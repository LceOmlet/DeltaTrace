"""Original-owner B4 joint/single EOS diagnostic; no training change.

Replay each original logged batch with its original dense EOS extension, target
positions, checkpoint and owner options. Single-deletion endpoints are evaluated
by that same runner, rather than another native batch/cache/layout. This observes
joint decomposition, single-input finite propagation, and native root differences
separately. All raw values are retained; no new acceptance threshold is imposed.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
import time

import torch


def source_of(owner):
    path = inspect.getsourcefile(inspect.unwrap(owner))
    from pathlib import Path
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def observe_matched_attribution(worker, out, records_path, source_path, runner_sha):
    from deltatrace_rollout import DeltaTraceRolloutProducer
    from deltatrace_credit import trace_token_attribution
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    from verify_textcraft_credit_degradation import memory_snapshot, selected_positions, sampled_qva

    entry, = [x for x in json.loads(records_path.read_bytes()) if x['step'] == 26]
    lines = json.loads(os.environ['DT_TEXTCRAFT_RECORD_LINES'])
    record, = [x for x in entry['minimum_records'] if x['line'] == lines[worker.rank]]
    samples = record['data']['samples']
    assert len(samples) == 4 and worker.actor.config.ppo_micro_batch_size_per_gpu == 4
    assert worker.config.model.lora_rank == 8 and worker.config.model.lora_alpha == 16
    eos = record['data']['eos_token_id']
    outcomes = record['data']['outcome_token_ids']
    probes = [selected_positions(s, worker.tokenizer) for s in samples]
    length, = set(s['trace']['compute_tokens'] for s in samples)
    # The original EventRatioReadout fills dense extension strictly after the
    # literal target with EOS, not the tokenizer pad ID. Keep its exact ABI.
    factual = torch.full((4, length), eos, dtype=torch.long, device='cuda')
    cases = []
    for i, sample in enumerate(samples):
        ids = sample['selected_input_ids']
        factual[i, :len(ids)] = torch.tensor(ids, device='cuda')
        cases.append(dict(target_ids=torch.tensor([ids[-1]]), prompt_length=len(ids)-1))
    result = dict(rank=worker.rank, pid=os.getpid(), original_step=26,
        checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'], original_record_line=record['line'],
        records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        paired_batch_size=8, local_source_batch_size=4, dense_compute_tokens=length,
        selection_scope='Original first minimum B4 per rank; selected extrema, not representative full batch',
        comparison_scope=__doc__, optimizer_steps=0, passes=[], samples=[])
    for sample, positions in zip(samples, probes):
        result['samples'].append(dict(traj_uid=sample['traj_uid'], source_start=sample['source_start'],
            source_end=sample['source_end'], original_context_tokens=len(sample['selected_input_ids']),
            original_selected_input_ids=sample['selected_input_ids'], observed_return=sample['observed_return'],
            original_trace=sample['trace'], original_source_signed=sample['source_signed'], probes=positions,
            probe_label_scope='v1 labels retained literally; actual token text is authoritative, including <think>'))
    path = out / f'rank{worker.rank}.json'
    def save(phase):
        result.update(phase=phase, observed_unix=time.time())
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(rank=worker.rank, phase=phase, passes=len(result['passes']))), flush=True)
    training = worker.actor_module_fsdp.training
    producer = runner = text = None
    previous_attention = None
    try:
        if worker._is_offload_param:
            load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        worker.actor_module_fsdp.eval()
        producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
            eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id,
            invalid_action_penalty_coef=(worker.config.actor.invalid_action_penalty_coef
                if worker.config.actor.get('use_invalid_action_penalty', True) else 0.0))
        runner = producer.runner
        PackedAnswerTargets = producer.packed_answer_targets
        result['imported_sources'] = dict(dt_runner=source_of(type(runner)),
            trace_token_attribution=source_of(trace_token_attribution),
            packed_answer_targets=source_of(PackedAnswerTargets),
            original_actor=source_of(type(worker.actor)))
        assert result['imported_sources']['dt_runner']['sha256'] == runner_sha
        assert runner.reuse_native_prefix
        text = runner.model.model.language_model
        previous_attention = text.config._attn_implementation
        text.set_attn_implementation('flash_attention_2')
        if producer.native_fla_fp16:
            from accelerated.qwen35.native_fla_precision import native_fla_fp16
            precision = native_fla_fp16(worker.actor_module_fsdp)
        else:
            precision = nullcontext()
        result.update(native_fla_fp16=producer.native_fla_fp16,
            numerical_runner_options={k: getattr(runner,k,None) for k in
                ('copy_replay_captures','offload_replay_mixer','gdn_head_batch_size',
                 'defer_diagnostics','pin_replay_host','pin_root_host','reuse_native_prefix')},
            allocation_before=memory_snapshot())
        save('original_worker_ready')
        with torch.no_grad(), precision:
            modes = (('single_v1_maximum_positive_unobserved_before',
                      'single_v1_maximum_positive_norm_observed',
                      'single_v1_maximum_positive_unobserved_after')
                if os.environ.get('DT_TEXTCRAFT_NORM_LEDGER') == '1' else
                ('single_v1_maximum_positive',)
                if (os.environ.get('DT_TEXTCRAFT_EFFECT_LEDGER') == '1' or
                    os.environ.get('DT_TEXTCRAFT_DECODER3_LEDGER') == '1') else
                ('original_joint_response','single_v1_negative_format','single_v1_maximum_positive'))
            for mode in modes:
                reference = factual.clone()
                for i, sample in enumerate(samples):
                    if mode == 'original_joint_response':
                        reference[i,sample['source_start']:sample['source_end']] = eos
                    else:
                        index = 0 if mode == 'single_v1_negative_format' else 1
                        reference[i,probes[i][index]['input_position']] = eos
                began = time.perf_counter()
                effects = None
                observation = nullcontext()
                if os.environ.get('DT_TEXTCRAFT_EFFECT_LEDGER') == '1':
                    from observe_textcraft_finite_effects import observe_token_effects
                    observation = observe_token_effects(runner)
                    result['imported_sources']['passive_effect_observer'] = source_of(observe_token_effects)
                elif os.environ.get('DT_TEXTCRAFT_DECODER3_LEDGER') == '1':
                    from observe_textcraft_layer3_finite_stages import observe_layer3_finite_stages
                    observation = observe_layer3_finite_stages(runner)
                    result['imported_sources']['passive_decoder_observer'] = source_of(observe_layer3_finite_stages)
                elif (os.environ.get('DT_TEXTCRAFT_NORM_LEDGER') == '1' and
                      mode.endswith('_norm_observed')):
                    from observe_textcraft_norm_operands import observe_norm_operands
                    observation = observe_norm_operands(runner, out / f'rank{worker.rank}-norm-operands.pt')
                    result['imported_sources']['passive_norm_observer'] = source_of(observe_norm_operands)
                with observation as effects:
                    signed, roots, detail = trace_token_attribution(runner, reference, factual,
                        cases, [[0] for _ in samples], packed_answer_targets=PackedAnswerTargets,
                        outcome_token_ids=outcomes)
                # Keep the owner ledger, native endpoint scores and all signed
                # entries. Conservation is a diagnostic, never token accuracy.
                values = []
                for i, sample in enumerate(samples):
                    vector = signed[i,sample['source_start']:sample['source_end']].cpu().tolist()
                    values.append(dict(traj_uid=sample['traj_uid'], source_signed=vector,
                        root_effect=float(roots[i]), detail=detail['per_sample'][i],
                        qva=sampled_qva(vector,sample['observed_return']),
                        probe_signed=[float(signed[i,p['input_position']]) for p in probes[i]],
                        all_signed_sum=float(signed[i].sum())))
                result['passes'].append(dict(mode=mode, seconds=time.perf_counter()-began,
                    owner_detail=detail, values=values, allocation=memory_snapshot(),
                    passive_finite_effects=(effects if os.environ.get('DT_TEXTCRAFT_EFFECT_LEDGER') == '1' else None),
                    passive_decoder_stages=(effects if os.environ.get('DT_TEXTCRAFT_DECODER3_LEDGER') == '1' else None),
                    passive_norm_operands=(effects if mode.endswith('_norm_observed') else None)))
                del signed, roots, reference
                save('original_owner_'+mode+'_completed')
        result['allocation_after'] = memory_snapshot()
        save('completed_matched_raw_diagnostic')
        return result
    finally:
        if text is not None and previous_attention is not None:
            text.set_attn_implementation(previous_attention)
        worker.actor_module_fsdp.train(training)
        if runner is not None:
            runner.model.release_owner_params()
        if worker._is_offload_param:
            offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
