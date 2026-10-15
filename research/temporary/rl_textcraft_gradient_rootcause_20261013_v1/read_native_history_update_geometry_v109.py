"""Read actual historical LoRA drift and saved native-update contrasts on CPU.

The historical gradient norm can grow without a comparable growth in Adam's
adaptive increment. Read both from existing native states, then compare the
already saved four-update interventions with the actual40-to80 parameter drift.
No model, gradient, optimizer step, training restart, DT or sampling is run.
"""
from collections import defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import time

import torch
import psutil
from verl.workers.fsdp_workers import ActorRolloutRefWorker

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT/'native-history-update-geometry-v109.json'
history_path = OUT/'temporal-adapter-states-v2.json'
original = json.loads(history_path.read_bytes())
manifest_path = OUT/'native-saved-update-natural-PG-v99/case-manifest.json'
manifest = json.loads(manifest_path.read_bytes())
worker_path = Path(inspect.getsourcefile(ActorRolloutRefWorker))
worker_lines = worker_path.read_text().splitlines()
worker_sha = hashlib.sha256(worker_path.read_bytes()).hexdigest()
assert worker_sha == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
constructor_lines = set()
for n, line in enumerate(worker_lines):
    if 'torch.optim.AdamW(' in line:
        constructor_lines.update(range(max(n-1, 0), min(n+12, len(worker_lines))))
report = dict(unix=time.time(), scope=__doc__, phase='reading_existing_native_states',
    pid=os.getpid(), birth=psutil.Process().create_time(),
    diagnostic_source=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    source_history=dict(path=str(history_path), SHA256=hashlib.sha256(history_path.read_bytes()).hexdigest()),
    source_saved_updates=dict(path=str(manifest_path), SHA256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()),
    worker_owner=dict(path=str(worker_path), SHA256=worker_sha,
        native_optimizer_constructor=[dict(line=n+1, text=worker_lines[n]) for n in sorted(constructor_lines)]),
    checkpoints=[], history_windows=[], saved_native_update_contrasts=[],
    new_model_calls=0, new_backward_calls=0, optimizer_steps=0, formal_training_changes=0,
    cuda_initialized=torch.cuda.is_initialized(), rootcause_repair_verified=False)


def save():
    report['unix'] = time.time()
    DEST.write_text(json.dumps(report, indent=2)+'\n')


def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def digest(values):
    h = hashlib.sha256()
    for name, value in values.items():
        h.update(name.encode())
        h.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def groups(name):
    layer = int(re.search(r'\.layers\.(\d+)\.', name)[1])
    side = 'A' if '.lora_A.' in name else 'B'
    return ['all', side, 'layers0_7' if layer < 8 else 'layers8_23' if layer < 24 else 'layers24_31',
            f'layer{layer}']


def vector_statistics(left, right, reference=None):
    assert list(left) == list(right)
    sums = defaultdict(lambda: dict(delta_sumsq=0., reference_sumsq=0., dot=0.))
    for name in left:
        delta = right[name].double() - left[name].double()
        ref = None if reference is None else reference[name].double()
        square = float(delta.square().sum())
        dot = 0. if ref is None else float((delta*ref).sum())
        rsq = 0. if ref is None else float(ref.square().sum())
        for key in groups(name):
            v = sums[key]
            v['delta_sumsq'] += square
            v['reference_sumsq'] += rsq
            v['dot'] += dot
    for row in sums.values():
        row['delta_norm'] = math.sqrt(row['delta_sumsq'])
        if reference is not None:
            row['reference_norm'] = math.sqrt(row['reference_sumsq'])
            den = row['delta_norm']*row['reference_norm']
            row['cosine_to_historical_drift'] = row['dot']/den if den else None
            row['projection_fraction_of_historical_drift'] = row['dot']/row['reference_sumsq'] if row['reference_sumsq'] else None
    return dict(sums)


saved_history = {0: {}, 1: {}}
previous = {}
for checkpoint in original['checkpoints']:
    step = checkpoint['step']
    entry = dict(iteration=step, optimizer_steps=[], ranks=[],
                 adaptive_increment_squared=0., weight_squared=dict(A=0., B=0.))
    for rr in checkpoint['ranks']:
        rank = rr['rank']
        mp, op = Path(rr['files']['model']), Path(rr['files']['optimizer'])
        model = torch.load(mp, map_location='cpu', weights_only=False, mmap=True)
        names = [k for k in model if '.lora_' in k]
        assert names == rr['LoRA_tensor_names'] and len(names) == 496
        values = {name: local(model[name]).detach().clone() for name in names}
        if step in (40, 80):
            chain_path = OUT/f'native-full-policy-direction-exchange-v86/iterationfull{step}_native-rank{rank}-chain.json'
            chain = json.loads(chain_path.read_bytes())
            assert digest(values) == chain['trainable_SHA256_before'] == chain['trainable_SHA256_after']
            assert names == [p['name'] for p in chain['native_accumulation_readout']['parameters']]
            saved_history[rank][step] = values
        optim = torch.load(op, map_location='cpu', weights_only=False, mmap=True)
        ids = [p for group in optim['param_groups'] for p in group['params'] if p in optim['state']]
        assert len(ids) == len(names) == len(optim['state'])
        assert len(optim['param_groups']) == 1
        group = optim['param_groups'][0]
        beta1, beta2 = group['betas']
        assert (beta1, beta2, group['eps'], group['lr'], group['weight_decay']) == (.9, .999, 1e-8, 1e-6, .01)
        incsq = 0.
        steps = set()
        for name, pid in zip(names, ids):
            state = optim['state'][pid]
            count = float(state['step'])
            steps.add(count)
            m, v = [local(state[k]).double() for k in ('exp_avg', 'exp_avg_sq')]
            assert m.shape == v.shape == values[name].shape
            assert bool((v >= 0).all())
            # A diagnostic norm of the native moment ratio, not an optimizer
            # implementation or a bit-exact reconstruction of a CUDA step.
            ratio = (m/(1-beta1**count)) / (v.sqrt()/math.sqrt(1-beta2**count)+group['eps'])
            incsq += float(ratio.square().sum())*group['lr']**2
            side = 'A' if '.lora_A.' in name else 'B'
            entry['weight_squared'][side] += float(values[name].double().square().sum())
        assert steps == {float(4*step)}
        entry['optimizer_steps'].append(4*step)
        entry['adaptive_increment_squared'] += incsq
        entry['ranks'].append(dict(rank=rank, trainable_SHA256=digest(values),
            model=str(mp), optimizer=str(op), native_trainable_names_equal_saved_order=True,
            adaptive_increment_norm=math.sqrt(incsq),
            optimizer_order_scope='Native saved parameter order, matched496 shapes; checked against actual named-parameter order at40/80 and pinned native AdamW constructor.'))
        if rank in previous:
            prevstep, prev = previous[rank]
            stats = vector_statistics(prev, values)
            window = next((x for x in report['history_windows'] if x['iterations'] == [prevstep, step]), None)
            if window is None:
                window = dict(iterations=[prevstep, step], optimizer_updates=4*(step-prevstep), ranks=[])
                report['history_windows'].append(window)
            window['ranks'].append(dict(rank=rank, statistics=stats))
        previous[rank] = (step, values)
        del model, optim
    entry['adaptive_increment_norm'] = math.sqrt(entry['adaptive_increment_squared'])
    report['checkpoints'].append(entry)
    save()

for window in report['history_windows']:
    square = math.fsum(r['statistics']['all']['delta_sumsq'] for r in window['ranks'])
    window.update(net_parameter_drift_norm=math.sqrt(square),
        net_drift_norm_per_update=math.sqrt(square)/window['optimizer_updates'],
        scope='Norm of net endpoint parameter displacement divided by updates; not mean step norm or RMS step norm.')

contrasts = defaultdict(list)
for rank in (0, 1):
    mid, base = saved_history[rank][40], saved_history[rank][80]
    reference = {name: base[name]-mid[name] for name in base}
    afters = {}
    for case in manifest['cases']:
        if case['name'] in ('old80_before', 'old80_before_repeat'):
            continue
        artifact = case['diagnostic_parameter_partition']['rank_artifacts'][str(rank)]
        path = Path(artifact['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['SHA256']
        raw = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        values = {name: local(value) for name, value in raw.items()}
        assert list(values) == list(base) and digest(values) == artifact['expected_result_trainable_SHA256']
        afters[case['name']] = values
        contrasts[case['name']].append(dict(rank=rank, statistics=vector_statistics(base, values, reference)))
    for label, left, right in [
        ('v47_original_minus_same_reward_ablated', 'v47_without_constant_positive_component_after', 'v47_full_original_after'),
        ('v69_original_minus_approved_v3', 'v69_approved_v3_full_method_after', 'v69_full_original_after')]:
        contrasts[label].append(dict(rank=rank,
            statistics=vector_statistics(afters[left], afters[right], reference)))

for label, ranks in contrasts.items():
    aggregate = {}
    for key in ranks[0]['statistics']:
        dsq = math.fsum(r['statistics'][key]['delta_sumsq'] for r in ranks)
        rsq = math.fsum(r['statistics'][key]['reference_sumsq'] for r in ranks)
        dot = math.fsum(r['statistics'][key]['dot'] for r in ranks)
        den = math.sqrt(dsq*rsq)
        aggregate[key] = dict(delta_norm=math.sqrt(dsq), historical_drift_norm=math.sqrt(rsq),
            cosine_to_historical_drift=dot/den if den else None,
            projection_fraction_of_historical_drift=dot/rsq if rsq else None)
    report['saved_native_update_contrasts'].append(dict(case=label, ranks=ranks, aggregate=aggregate))
report.update(phase='complete_CPU_native_history_and_saved_update_geometry',
    adaptive_increment_scope='lr*||m_hat/(sqrt(v_hat)+native_epsilon)|| from saved native moments, evaluated in FP64. This is the adaptive term of the mathematical just-completed AdamW step, excluding weight decay/native finite precision; it is not the next update, a new optimizer, or an exact recovered CUDA parameter change.',
    limits=[
        'Four-update intervention contrasts are actual saved native outcomes, not linear gradient component contributions or the full historical trajectory.',
        'A direction cosine with endpoint drift cannot by itself establish temporal causation or explain the nonlinear propagation gain.',
        'History windows are unequal. Net displacement/update can hide cancellation and cannot be called per-step update RMS.',
        'Adam moments contain joint clipped actor gradients; they do not separate policy, entropy and KL gradients.',
        'No new mathematical tolerance, official implementation replacement, from-scratch outcome, training-quality claim or root repair is introduced.',
    ])
assert not torch.cuda.is_initialized()
save()
print(json.dumps(dict(receipt=str(DEST), report=report)))
