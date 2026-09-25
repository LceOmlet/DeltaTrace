"""Bounded same-actor route/precision audit, not a new credit estimator.

All finite calls use the real RL boundary. Native measurements use the same
paired B8 IDs, target positions and original HF forward, not a shorter B4
prefix. Precision scopes are the existing tested native FLA option. No new
acceptance threshold is imposed on the whole-network signed vector.
"""
import json
import os
from contextlib import nullcontext
from pathlib import Path
import time

import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from deltatrace_credit import trace_token_attribution
from qwen35_answer_finite import PackedAnswerTargets, categorical_head_logits
from accelerated.qwen35.native_fla_precision import native_fla_fp16


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    root = Path(os.environ['DT_RUNTIME_ROOT'])
    source = Path(os.environ.get('CREDIT_ROUTE_SOURCE', str(
        root/'receipts/dt-extremes-side-20260925-1027/fixed-input-trace-0.pt')))
    output = Path(os.environ['CREDIT_ROUTE_OUTPUT'])
    payload = json.loads(source.read_text()) if source.suffix == '.json' else None
    saved = None if payload is not None else torch.load(source, map_location='cpu', weights_only=True)
    report = dict(scope=__doc__, source=str(source), runs=[], method_changed=False)
    started = time.perf_counter()
    def save(phase):
        report.update(phase=phase, seconds=time.perf_counter()-started)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, seconds=report['seconds'])), flush=True)

    save('original_actor_init')
    worker = ActorRolloutRefWorker(configuration(1024).actor_rollout_ref, 'actor')
    worker.init_model()
    producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp.eval(),
        eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
    runner = producer.runner
    text_model = runner.model.model.language_model
    attention = text_model.config._attn_implementation
    text_model.set_attn_implementation('flash_attention_2')
    labels = producer.readout.alphabet.label_ids(worker.tokenizer)
    if payload is None:
        pair = saved['pair'].cuda()
        cases = [dict(target_ids=row[-1:].cpu(), prompt_length=len(row)-1) for row in pair[1::2]]
    else:
        samples = payload['samples']
        assert len(samples) == 4 and labels == payload['outcome_token_ids']
        assert payload['eos_token_id'] == worker.tokenizer.eos_token_id
        length = max(sample['trace']['compute_tokens'] for sample in samples)
        pair = torch.full((8, length), payload['eos_token_id'], dtype=torch.long, device='cuda')
        cases = []
        for b, sample in enumerate(samples):
            ids = torch.tensor(sample['selected_input_ids'], dtype=torch.long)
            assert len(ids) == sample['trace']['context_tokens']
            pair[2*b:2*b+2, :len(ids)] = ids.to(pair.device)
            pair[2*b, sample['source_start']:sample['source_end']] = payload['eos_token_id']
            cases.append(dict(target_ids=ids[-1:], prompt_length=len(ids)-1))
        report['replay_scope'] = ('Original saved token IDs, event labels and batch padding; '
            'fresh initial actor, not restoration of the former worker process.')
    selection = PackedAnswerTargets(cases, [[0]]*4, pair.shape[1], pair.device, outcome_token_ids=labels)
    from native_target_logit_rows import NativeTargetLogitRows
    selector = NativeTargetLogitRows(selection)
    categories = (selection.labels[:, None] == selection.outcome_token_ids[None]).long().argmax(1)
    changed = pair[0::2] != pair[1::2]
    first = changed.int().argmax(1).tolist()
    if payload is not None:
        first = [sample['source_start'] + sample['trace']['source_log_ratio_min_index']
                 for sample in payload['samples']]
    single = pair.clone()
    single[0::2] = single[1::2]
    for b, position in enumerate(first):
        single[2*b, position] = worker.tokenizer.eos_token_id
    prefix = runner.reuse_native_prefix
    artifacts = dict(pair=pair.cpu(), single_pair=single.cpu(), first_positions=first, calls={})

    def native(paired):
        values = []
        def capture(module, args):
            # Use the existing target-row owner for distinct predictor positions.
            # Projection stays inside the gathered FSDP head lifetime.
            logits = categorical_head_logits(module, args[0], selection.outcome_token_ids)
            values.append(selector.pack_logits(logits).detach())
        handle = runner.model.lm_head.register_forward_pre_hook(capture)
        try:
            runner.model(input_ids=paired, attention_mask=torch.ones_like(paired),
                         use_cache=False, logits_to_keep=selector.rows)
        finally:
            handle.remove()
            runner.model.release_owner_params()
        (logits,) = values
        assert logits.shape == (8, len(labels))
        logp = logits.log_softmax(-1).gather(1, categories.repeat_interleave(2)[:, None]).squeeze(1)
        return dict(logp0=logp[0::2].cpu().tolist(), logp1=logp[1::2].cpu().tolist(),
                    root=(logp[1::2]-logp[0::2]).cpu().tolist())

    try:
        for precision in (('native_fla_fp16',) if payload is not None else ('native_bf16', 'native_fla_fp16')):
            context = nullcontext() if precision == 'native_bf16' else native_fla_fp16(producer.actor)
            with context:
                save(precision+'_native_paired_start')
                refs = dict(joint=native(pair), single=native(single), single_repeat=native(single))
                report.setdefault('native', {})[precision] = refs
                save(precision+'_native_paired_done')
                schedule = [('joint', pair, True), ('joint_repeat', pair, True), ('single', single, True)]
                if payload is not None:
                    schedule = [('joint', pair, True), ('single', single, True)]
                if precision == 'native_fla_fp16':
                    if os.environ.get('CREDIT_ROUTE_AUDIT') == '1':
                        schedule.append(('single_repeat', single, True))
                    schedule.append(('joint_full', pair, False))
                for name, inputs, reuse in schedule:
                    runner.reuse_native_prefix = reuse
                    label = precision+'_'+name
                    save(label+'_start')
                    tick = time.perf_counter()
                    audits = []
                    audit = nullcontext()
                    if os.environ.get('CREDIT_ROUTE_AUDIT') == '1' and precision == 'native_fla_fp16' and name == 'single':
                        from audit_finite_residuals import boundary_audit
                        selected_layers = range(32) if os.environ.get('CREDIT_ROUTE_AUDIT_ALL') == '1' else (13, 7, 8)
                        audit = boundary_audit(runner, audits, selected=selected_layers)
                    with audit:
                        signed, roots, detail = trace_token_attribution(runner, inputs[0::2], inputs[1::2],
                            cases, [[0]]*4, packed_answer_targets=PackedAnswerTargets, outcome_token_ids=labels)
                    # The public runner returns CPU attribution. Persist its
                    # result before doing optional diagnostic post-processing.
                    signed = signed.cpu()
                    artifacts['calls'][label] = dict(signed=signed, roots=roots.cpu(), detail=detail)
                    torch.save(artifacts, output.with_suffix('.pt'))
                    active = changed.cpu()
                    rows = []
                    for b in range(4):
                        values = signed[b, active[b]].double()
                        lp = detail['target_logp1'][b]
                        rows.append(dict(event=payload['samples'][b]['trace']['event_step'] if payload else 10+b,
                            root=float(roots[b]), factual_logp=lp,
                            reference_logp=detail['target_logp0'][b], selected_position=first[b],
                            selected_signed=float(signed[b, first[b]]),
                            d_min=float(values.min()), d_max=float(values.max()), signed_sum=float(values.sum()),
                            max_implied_probability=float((lp-values).exp().max())))
                        if payload is not None:
                            sample = payload['samples'][b]
                            recorded = torch.tensor(sample['source_signed'], dtype=torch.float64)
                            replayed = signed[b, sample['source_start']:sample['source_end']].double()
                            rows[-1]['saved_joint_d_min'] = float(recorded.min())
                            if name == 'joint':
                                rows[-1]['saved_joint_max_abs_difference'] = float((recorded-replayed).abs().max())
                    run = dict(precision=precision, name=name, native_prefix=reuse,
                               seconds=time.perf_counter()-tick, rows=rows)
                    if audits:
                        run['boundary_audit'] = audits
                    report['runs'].append(run)
                    save(label+'_done')
        save('completed')
    except BaseException as exc:
        report['error'] = repr(exc)
        save('failed')
        raise
    finally:
        runner.reuse_native_prefix = prefix
        text_model.set_attn_implementation(attention)
        runner.model.release_owner_params()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
