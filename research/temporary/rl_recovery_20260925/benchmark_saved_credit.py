"""Compare complete DT with the original actor backward on exact saved IDs.

The original VERL worker owns vLLM residency, FSDP, activation offload and
checkpointing. No optimizer update, environment rollout or alternate backward
is implemented. Native backward keeps its original BF16 output logits; DT
uses the repaired FP32 event readout. This is a cost comparison, not a claim
that the two algorithms or their gradients are numerically identical.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from deltatrace_credit import trace_token_attribution
from qwen35_answer_finite import PackedAnswerTargets
from accelerated.qwen35.native_fla_precision import native_fla_fp16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parameter-offload-policy', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    payload = json.loads(args.source.read_text())
    report = dict(scope=__doc__, source=str(args.source),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  context_cap=32768, batch=4, activation_offload=True,
                  native_gradient_checkpointing=True,
                  parameter_offload_policy=args.parameter_offload_policy,
                  vllm_max_num_seqs=32, runs=[])
    started = time.perf_counter()
    text_model = None
    original_attention = None

    def save(phase):
        report.update(phase=phase, elapsed=time.perf_counter()-started)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, elapsed=report['elapsed'])), flush=True)

    try:
        save('native_worker_init')
        cfg = configuration(1024).actor_rollout_ref
        cfg.actor.fsdp_config.offload_policy = args.parameter_offload_policy
        cfg.model.enable_gradient_checkpointing = True
        cfg.rollout.enforce_eager = False
        cfg.rollout.free_cache_engine = False
        cfg.rollout.engine_kwargs.vllm.enable_prefix_caching = False
        worker = ActorRolloutRefWorker(cfg, 'actor_rollout')
        worker.init_model()
        with worker.rollout_sharding_manager:
            pass
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
        if worker._is_offload_param:
            load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        actor = worker.actor_module_fsdp
        producer = DeltaTraceRolloutProducer(actor.eval(), eos_token_id=worker.tokenizer.eos_token_id,
                                            pad_token_id=worker.tokenizer.pad_token_id)
        # Match the existing producer's scoped DT attention selection, then
        # restore the owner's backend before measuring its original backward.
        text_model = producer.runner.model.model.language_model
        original_attention = text_model.config._attn_implementation
        samples = payload['samples']
        labels = payload['outcome_token_ids']
        assert len(samples) == 4 and labels == producer.readout.alphabet.label_ids(worker.tokenizer)
        assert payload['eos_token_id'] == worker.tokenizer.eos_token_id
        length = max(row['trace']['compute_tokens'] for row in samples)
        selected = torch.full((4, length), payload['eos_token_id'], device='cuda', dtype=torch.long)
        reference = selected.clone()
        cases, positions, categories = [], [], []
        for b, row in enumerate(samples):
            ids = torch.tensor(row['selected_input_ids'], dtype=torch.long)
            assert len(ids) == row['trace']['context_tokens']
            selected[b, :len(ids)] = ids.to(selected.device)
            reference[b] = selected[b]
            reference[b, row['source_start']:row['source_end']] = payload['eos_token_id']
            cases.append(dict(target_ids=ids[-1:], prompt_length=len(ids)-1))
            positions.append(len(ids)-2)
            categories.append(labels.index(int(ids[-1])))
        report.update(tokens=length, target_predictor_positions=positions,
                      factual_ids_sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest())
        save('resident_vllm_and_actor_ready')
        text_model.set_attn_implementation('flash_attention_2')
        for repeat in range(2):
            save('complete_dt_'+str(repeat))
            actor.eval()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.no_grad(), native_fla_fp16(actor):
                signed, roots, detail = trace_token_attribution(
                    producer.runner, reference, selected, cases, [[0]]*4,
                    packed_answer_targets=PackedAnswerTargets, outcome_token_ids=labels)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            groups = defaultdict(float)
            for call in detail['calls']:
                import re
                groups[re.sub(r'_\d+$', '', call['kind'])] += call['stream_elapsed_seconds']
            report['runs'].append(dict(kind='complete_dt', repeat=repeat, seconds=seconds,
                phase_stream_seconds=dict(groups), d_min=float(signed.min()), d_max=float(signed.max()),
                full_detail=detail))
            save('complete_dt_'+str(repeat)+'_done')
        text_model.set_attn_implementation(original_attention)
        predictor, inverse = torch.unique(torch.tensor(positions, device='cuda'), sorted=True, return_inverse=True)
        label_ids = torch.tensor(labels, device='cuda')
        category_ids = torch.tensor(categories, device='cuda')
        for repeat in range(2):
            save('native_backward_'+str(repeat))
            actor.train()
            actor.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output = actor(input_ids=selected, attention_mask=torch.ones_like(selected),
                           use_cache=False, logits_to_keep=predictor)
            assert output.logits.shape[:2] == (4, len(predictor))
            logits = output.logits[torch.arange(4, device='cuda'), inverse]
            logp = logits.index_select(-1, label_ids).float().log_softmax(-1)
            loss = -logp.gather(1, category_ids[:, None]).sum()
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter()-tick
            gradients = [p.grad for p in actor.parameters() if p.requires_grad and p.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(
                g.to_local() if hasattr(g, 'to_local') else g).all()) for g in gradients)
            assert finite
            report['runs'].append(dict(kind='native_backward', repeat=repeat,
                forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                finite_gradients=finite, loss=float(loss.detach())))
            del output, logits, logp, loss, gradients
            save('native_backward_'+str(repeat)+'_done')
        actor.zero_grad(set_to_none=True)
        report['physical_gpu_snapshot'] = subprocess.run(['mx-smi'], capture_output=True, text=True).stdout
        save('completed')
    except BaseException as exc:
        report['error'] = repr(exc)
        save('failed')
        raise
    finally:
        if text_model is not None:
            text_model.set_attn_implementation(original_attention)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
