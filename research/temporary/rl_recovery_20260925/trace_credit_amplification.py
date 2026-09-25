"""Passive, bounded audit of the actual finite owners on a saved B4 pair.

No observer flag, altered prefix setting, new finite rule, or training gate.
Old and repaired heads share one actor and all body settings. Single-token
native interventions below are diagnostics only, never the production method.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from qwen35_answer_finite import PackedAnswerTargets


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(value):
    if isinstance(value, (tuple, list)):
        return [stats(v) for v in value]
    a = value.detach().float().flatten(1)
    return dict(absmax=a.abs().amax(1).cpu().tolist(),
                l2=a.square().sum(1).sqrt().cpu().tolist())


def effect_stats(owner, coeff, endpoints):
    v = owner._token_effect(coeff, endpoints)
    return dict(min=v.min(1).values.cpu().tolist(), max=v.max(1).values.cpu().tolist(),
                sum=v.sum(1).cpu().tolist(), abs_sum=v.abs().sum(1).cpu().tolist(),
                min_position=v.argmin(1).cpu().tolist())


def scalar_secant_stats(x0, x1, y0, y1):
    x0, x1, y0, y1 = [v.detach().float() for v in (x0, x1, y0, y1)]
    dx, dy = x1-x0, y1-y0
    nonzero = dx != 0
    s = torch.where(nonzero, dy/torch.where(nonzero, dx, 1), 0).flatten(1)
    index = s.abs().argmax(1, keepdim=True)
    return dict(absmax=s.abs().amax(1).cpu().tolist(),
                numerator=dy.flatten(1).gather(1, index).flatten().cpu().tolist(),
                denominator=dx.flatten(1).gather(1, index).flatten().cpu().tolist(),
                input0=x0.flatten(1).gather(1, index).flatten().cpu().tolist(),
                equal_input_unequal_output=((~nonzero)&(dy != 0)).flatten(1).sum(1).cpu().tolist())


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    root = Path(os.environ['DT_RUNTIME_ROOT'])
    output = Path(os.environ['CREDIT_TRACE_OUTPUT'])
    source = Path(os.environ.get('CREDIT_TRACE_SOURCE', str(root/'receipts/dt-extremes-side-20260925-1027/fixed-input-trace-0.pt')))
    saved = torch.load(source, map_location='cpu', weights_only=True)
    report = dict(scope=__doc__, source=str(source), runs=[])
    resume = os.environ.get('CREDIT_TRACE_RESUME')
    if resume:
        report = json.loads(Path(resume).read_text())
        report['resumed_diagnostic'] = dict(source=resume, prior_error=report.pop('error', None))
    started = time.perf_counter()
    def save(phase):
        report.update(phase=phase, elapsed=time.perf_counter()-started)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, elapsed=report['elapsed'])), flush=True)

    save('load_original_actor')
    worker = ActorRolloutRefWorker(configuration(1024).actor_rollout_ref, 'actor')
    worker.init_model()
    producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp.eval(),
        eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
    runner = producer.runner
    text_model = runner.model.model.language_model
    attention = text_model.config._attn_implementation
    text_model.set_attn_implementation('flash_attention_2')
    controller = sys.modules[type(runner).__module__]
    oldpath = root/'releases/2ee7ab4/clean/qwen35'
    oldanswer = load('amplification_old_answer', oldpath/'qwen35_answer_finite.py')
    oldcontroller = load('amplification_old_controller', oldpath/'qwen35_dense_finite_runner.py')
    baseline = copy.copy(runner)
    baseline.answer = oldanswer.FiniteAnswerOps(compiled=False)
    pair = saved['pair'].to('cuda')
    labels = producer.readout.alphabet.label_ids(worker.tokenizer)
    label_tensor = torch.tensor(labels, device=pair.device)
    categories = torch.tensor([labels.index(int(row[-1])) for row in pair[1::2]], device=pair.device)
    targets = PackedAnswerTargets([dict(target_ids=row[-1:].cpu(), prompt_length=len(row)-1)
        for row in pair[1::2]], [[0]]*4, pair.shape[1], pair.device, outcome_token_ids=labels)
    changed = (pair[0::2] != pair[1::2]).cpu()
    layers = runner.model.model.language_model.layers
    layer_ids = {id(v): i for i, v in enumerate(layers)}
    artifacts = {}
    if resume:
        artifacts = torch.load(Path(resume).with_suffix('.pt'), map_location='cpu', weights_only=True)
    try:
        schedule = [] if resume else [('old_head', baseline, oldcontroller), ('fp32_head', runner, controller)]
        for name, owner, module in schedule:
            entry = dict(name=name, boundaries=[], layers=[])
            report['runs'].append(entry)
            context = dict(layer=32)
            originals = {}
            original_decoder = module.decoder_finite_pullback
            head_name = 'rounded_seed' if name == 'old_head' else 'categorical_seed'
            original_head = getattr(owner.answer, head_name)
            heads = []
            def head_logged(*args, **kwargs):
                result = original_head(*args, **kwargs)
                heads.append([v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in args])
                entry['head_coefficients'] = stats(result[0])
                return result
            def decoder_logged(layer, values, upstream, *args, **kwargs):
                context['layer'] = layer_ids[id(layer)]
                inp = values['input_norm_input']
                before = stats(upstream)
                result = original_decoder(layer, values, upstream, *args, **kwargs)
                entry['layers'].append(dict(layer=context['layer'], kind=layer.block_type,
                    incoming=before, outgoing=stats(result[0]),
                    effect=effect_stats(module, result[0], inp)))
                print(json.dumps(dict(run=name, **entry['layers'][-1])), flush=True)
                return result
            def wrap(boundary, fn):
                def logged(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    row = dict(layer=context['layer'], name=boundary, output=stats(result))
                    if boundary == 'mlp':
                        row['silu_secant'] = scalar_secant_stats(*args[:2], *args[4:6])
                    elif boundary == 'gdn_conv_silu':
                        pre, out = args[:2]
                        row['silu_secant'] = scalar_secant_stats(pre[0::2], pre[1::2], out[0::2], out[1::2])
                    elif boundary == 'norm_residual':
                        row['effect'] = effect_stats(module, result, torch.stack(args[:2], 1).flatten(0, 1))
                        row['upstream'] = stats(args[3])
                        row['residual'] = stats(args[4])
                    entry['boundaries'].append(row)
                    return result
                return logged
            for boundary in ('mlp', 'norm_residual', 'attention_gate', 'attention_input',
                             'gdn_norm_gate', 'gdn_conv_silu'):
                originals[boundary] = getattr(owner.boundaries, boundary)
                setattr(owner.boundaries, boundary, wrap(boundary, originals[boundary]))
            setattr(owner.answer, head_name, head_logged)
            module.decoder_finite_pullback = decoder_logged
            save(name+'_start')
            tick = time.perf_counter()
            try:
                signed, detail = module.Qwen35DenseFiniteRunner.attribute(
                    owner, pair, torch.ones_like(pair), targets, select_output_rows=True)
            finally:
                module.decoder_finite_pullback = original_decoder
                setattr(owner.answer, head_name, original_head)
                for boundary, fn in originals.items():
                    setattr(owner.boundaries, boundary, fn)
            rows = []
            for b in range(4):
                d = signed[b, changed[b]].double()
                lp = detail['target_logp1'][b]
                event = saved['cases'][b]['event'] if 'cases' in saved else b+10
                rows.append(dict(event=event, logp=lp, d_min=float(d.min()), d_max=float(d.max()),
                    sum=float(d.sum()), root=lp-detail['target_logp0'][b],
                    max_implied_p=float((lp-d).exp().max()), min_position=int(d.argmin())))
            entry.update(rows=rows, seconds=time.perf_counter()-tick)
            artifacts[name] = dict(signed=signed.cpu(), detail=detail, head=heads)
            torch.save(artifacts, output.with_suffix('.pt'))
            if name == 'old_head' and 'head' in saved:
                entry['head_inputs_equal_saved'] = all(torch.equal(a, b) for a, b in zip(heads[0], saved['head']))
                entry['head_operand_maxdiff'] = [float((a.double()-b.double()).abs().max())
                    for a, b in zip(heads[0], saved['head'])]
            save(name+'_done')
        # Actual model measurement of the most negative source token only.
        factual = pair[1::2, :-1].clone()
        deleted = factual.clone()
        positions = []
        for b, row in enumerate(report['runs'][-1]['rows']):
            absolute = changed[b].nonzero().flatten()[row['min_position']].item()
            positions.append(absolute)
            deleted[b, absolute] = worker.tokenizer.eos_token_id
        p = runner.read_outcomes(factual, label_tensor).gather(1, categories[:, None]).squeeze(1)
        q = runner.read_outcomes(deleted, label_tensor).gather(1, categories[:, None]).squeeze(1)
        report['single_eos_native'] = dict(positions=positions, logp=p.cpu().tolist(),
            deleted_logp=q.cpu().tolist(), difference=(p-q).cpu().tolist())
        first_deleted = factual.clone()
        first_positions = changed.to(torch.int32).argmax(1).tolist()
        for b, position in enumerate(first_positions):
            first_deleted[b, position] = worker.tokenizer.eos_token_id
        first_q = runner.read_outcomes(first_deleted, label_tensor).gather(1, categories[:, None]).squeeze(1)
        report['first_token_native'] = dict(positions=first_positions, logp=p.cpu().tolist(),
            deleted_logp=first_q.cpu().tolist(), difference=(p-first_q).cpu().tolist(),
            dt=[float(artifacts['fp32_head']['signed'][b, pos]) for b, pos in enumerate(first_positions)])
        # A B4 single-position DT call preserves the original packed/cache
        # evaluation route; its native root is an independent deletion check.
        single_pair = pair.clone()
        single_pair[0::2] = single_pair[1::2]
        for b, position in enumerate(first_positions):
            single_pair[2*b, position] = worker.tokenizer.eos_token_id
        single_signed, single_detail = runner.attribute(
            single_pair, torch.ones_like(single_pair), targets, select_output_rows=True)
        report['single_first_token_same_owner'] = dict(
            logp=single_detail['target_logp1'], deleted_logp=single_detail['target_logp0'],
            root=[a-b for a,b in zip(single_detail['target_logp1'],single_detail['target_logp0'])],
            dt=[float(single_signed[b,pos]) for b,pos in enumerate(first_positions)])
        save('completed')
    except BaseException as e:
        report['error'] = repr(e)
        save('failed')
        raise
    finally:
        text_model.set_attn_implementation(attention)
        runner.model.release_owner_params()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
