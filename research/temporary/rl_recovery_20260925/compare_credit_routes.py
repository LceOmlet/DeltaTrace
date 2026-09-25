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
    source = root/'receipts/dt-extremes-side-20260925-1027/fixed-input-trace-0.pt'
    output = Path(os.environ['CREDIT_ROUTE_OUTPUT'])
    saved = torch.load(source, map_location='cpu', weights_only=True)
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
    pair = saved['pair'].cuda()
    labels = producer.readout.alphabet.label_ids(worker.tokenizer)
    cases = [dict(target_ids=row[-1:].cpu(), prompt_length=len(row)-1) for row in pair[1::2]]
    selection = PackedAnswerTargets(cases, [[0]]*4, pair.shape[1], pair.device, outcome_token_ids=labels)
    categories = (selection.labels[:, None] == selection.outcome_token_ids[None]).long().argmax(1)
    changed = pair[0::2] != pair[1::2]
    first = changed.int().argmax(1).tolist()
    single = pair.clone()
    single[0::2] = single[1::2]
    for b, position in enumerate(first):
        single[2*b, position] = worker.tokenizer.eos_token_id
    prefix = runner.reuse_native_prefix
    artifacts = dict(pair=pair.cpu(), single_pair=single.cpu(), first_positions=first, calls={})

    def native(paired):
        values = []
        def capture(module, args):
            packed = args[0].reshape(-1, args[0].shape[-1])
            values.append(categorical_head_logits(module, packed, selection.outcome_token_ids).detach())
        handle = runner.model.lm_head.register_forward_pre_hook(capture)
        try:
            runner.model(input_ids=paired, attention_mask=torch.ones_like(paired),
                         use_cache=False, logits_to_keep=selection.positions.unique())
        finally:
            handle.remove()
            runner.model.release_owner_params()
        (logits,) = values
        assert logits.shape == (8, len(labels))
        logp = logits.log_softmax(-1).gather(1, categories.repeat_interleave(2)[:, None]).squeeze(1)
        return dict(logp0=logp[0::2].cpu().tolist(), logp1=logp[1::2].cpu().tolist(),
                    root=(logp[1::2]-logp[0::2]).cpu().tolist())

    try:
        for precision in ('native_bf16', 'native_fla_fp16'):
            context = nullcontext() if precision == 'native_bf16' else native_fla_fp16(producer.actor)
            with context:
                save(precision+'_native_paired_start')
                refs = dict(joint=native(pair), single=native(single), single_repeat=native(single))
                report.setdefault('native', {})[precision] = refs
                save(precision+'_native_paired_done')
                schedule = [('joint', pair, True), ('joint_repeat', pair, True), ('single', single, True)]
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
                        rows.append(dict(event=10+b, root=float(roots[b]), factual_logp=lp,
                            reference_logp=detail['target_logp0'][b], first_signed=float(signed[b, first[b]]),
                            d_min=float(values.min()), d_max=float(values.max()), signed_sum=float(values.sum()),
                            max_implied_probability=float((lp-values).exp().max())))
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
