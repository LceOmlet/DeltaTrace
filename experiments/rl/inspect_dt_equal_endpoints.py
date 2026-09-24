"""Diagnose native endpoint differences without replacing any owner computation.

Uses recorded owner prompt IDs and an explicitly synthetic EOS-only response.
No environment reward, task score, PPO update or new numerical tolerance is used.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
from deltatrace_rollout import DeltaTraceRolloutProducer
from deltatrace_credit import trace_token_attribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope=__doc__, status='running', observations=[])
    started = time.perf_counter()

    def save():
        result['elapsed'] = time.perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    try:
        torch.manual_seed(2026)
        worker = ActorRolloutRefWorker(configuration(1024).actor_rollout_ref, 'actor_rollout')
        worker.init_model()
        with worker.rollout_sharding_manager:
            pass
        if worker._is_offload_param:
            load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
            eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
        result['model_ready_seconds'] = time.perf_counter() - started
        save()
        prompt = json.loads(args.request.read_text())[0]['prompt_ids']
        tokenizer = worker.tokenizer
        alphabet = producer.readout.alphabet
        labels = alphabet.label_ids(tokenizer)
        actor = worker.actor_module_fsdp
        actor.eval()
        text_model = producer.runner.model.model.language_model
        previous_attention = text_model.config._attn_implementation
        text_model.set_attn_implementation('flash_attention_2')
        from qwen35_answer_finite import selected_target_log_probs
        try:
            for mixed in (False, True):
                rows, cases = [], []
                for event in range(1, 5):
                    query = alphabet.query_ids(tokenizer, current_step=1, event_step=event, max_steps=15)
                    ids = prompt + [tokenizer.eos_token_id] + query + [labels[0]]
                    rows.append(ids)
                    cases.append(dict(target_ids=torch.tensor([labels[0]]), prompt_length=len(ids)-1))
                selected = torch.tensor(rows, device='cuda')
                reference = selected.clone()
                if mixed:
                    selected[-1, len(prompt)] = tokenizer.encode(' up', add_special_tokens=False)[0]
                observation = dict(mixed=mixed, length=selected.shape[1],
                    endpoint_tokens_different=(selected != reference).sum(-1).tolist(), layer_calls=[])
                result['observations'].append(observation)
                handles = []

                def hook(index):
                    def observe(module, inputs, output):
                        value = output[0] if isinstance(output, tuple) else output
                        if value.shape[0] == 8:
                            # Record native values only; never alter the result.
                            delta = (value[0::2].float()-value[1::2].float()).abs()
                            observation['layer_calls'].append(dict(layer=index,
                                shape=list(value.shape), pair_max_delta=delta.flatten(1).amax(1).tolist()))
                    return observe

                for index, layer in enumerate(text_model.layers):
                    handles.append(layer.register_forward_hook(hook(index)))
                tick = time.perf_counter()
                try:
                    # First observe the original uncached full forward with
                    # the exact same endpoints and target rows. This isolates
                    # native low-precision differences from DT prefix reuse.
                    pair = torch.stack((reference, selected), dim=1).flatten(0, 1)
                    selection = producer.packed_answer_targets(
                        cases, [[0]]*4, pair.shape[1], pair.device, outcome_token_ids=labels)
                    with torch.no_grad():
                        native = producer.runner.model.forward_root(
                            input_ids=pair, attention_mask=torch.ones_like(pair),
                            use_cache=False, logits_to_keep=selection.positions.unique(sorted=True))
                        # All four one-token targets have the same predictor.
                        assert selection.positions.unique().numel() == 1
                        lp = selected_target_log_probs(native.logits[:, 0], selection)
                    observation['native_uncached_roots'] = (lp[1::2]-lp[0::2]).tolist()
                    observation['native_uncached_layer_calls'] = observation['layer_calls']
                    observation['layer_calls'] = []
                    save()
                    del native
                    signed, roots, detail = trace_token_attribution(
                        producer.runner, reference, selected, cases, [[0]]*4,
                        packed_answer_targets=producer.packed_answer_targets, outcome_token_ids=labels)
                    observation.update(seconds=time.perf_counter()-tick,
                        signed_max=signed.abs().amax(1).tolist(), roots=roots.tolist(), detail=detail)
                finally:
                    for handle in handles:
                        handle.remove()
                save()
                print(json.dumps({k:v for k,v in observation.items()
                                  if k not in ('detail','layer_calls','native_uncached_layer_calls')}), flush=True)
        finally:
            text_model.set_attn_implementation(previous_attention)
        result['status'] = 'completed_diagnostic_not_acceptance'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
