"""Account saved native LoRA updates in effective-weight space on CPU.

Raw A/B coordinates can change while their product is unchanged. Compare the
actual alpha/r * B A products using small Gram identities. This does not create
a LoRA forward/merge implementation or execute any model or optimizer.
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

import psutil
import torch
from peft.tuners.lora.layer import Linear
from verl.workers.fsdp_workers import ActorRolloutRefWorker

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
manifest_path = OUT/'native-saved-update-natural-PG-v99/case-manifest.json'
manifest = json.loads(manifest_path.read_bytes())
history_path = OUT/'temporal-adapter-states-v2.json'
history = json.loads(history_path.read_bytes())
native40 = next(c for c in history['checkpoints'] if c['step'] == 40)
scale = 2.0


def owner_lines(obj, words, radius=2):
    path = Path(inspect.getsourcefile(obj))
    lines = path.read_text().splitlines()
    chosen = set()
    for n, text in enumerate(lines):
        if any(w in text for w in words):
            chosen.update(range(max(0, n-radius), min(len(lines), n+radius+1)))
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                lines=[dict(line=n+1, text=lines[n]) for n in sorted(chosen)])


worker = owner_lines(ActorRolloutRefWorker, ['LoraConfig(', "'lora_alpha':", "'r': self.config.model.lora_rank"])
assert worker['SHA256'] == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
peft = owner_lines(Linear, ['def get_delta_weight', 'weight_B @ weight_A'])
launch = json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
assert launch['options']['actor_rollout_ref.model.lora_rank'] == 8
assert launch['options']['actor_rollout_ref.model.lora_alpha'] == 16
assert any("'lora_alpha': self.config.model.lora_alpha" in r['text'] for r in worker['lines'])
report = dict(unix=time.time(), phase='reading_saved_effective_adapter_updates', scope=__doc__,
    pid=os.getpid(), birth=psutil.Process().create_time(),
    diagnostic_source=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    LoRA_rank=8, LoRA_alpha=16, scaling=scale, native_worker_owner=worker, native_PEFT_owner=peft,
    source_saved_updates=dict(path=str(manifest_path), SHA256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()),
    sources=[], statistics=[], model_calls=0, backward_calls=0, optimizer_updates=0,
    formal_training_changes=0, rootcause_repair_verified=False)
dest = OUT/'effective-adapter-update-geometry-v112.json'


def save():
    report['unix'] = time.time()
    dest.write_text(json.dumps(report, indent=2)+'\n')


def local(x):
    return x.to_local() if hasattr(x, 'to_local') else x


def digest(values):
    h = hashlib.sha256()
    for name, value in values.items():
        h.update(name.encode())
        h.update(local(value).detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


states = {}
for name in ['old40', 'old80', 'v47_full_original_after',
             'v47_without_constant_positive_component_after',
             'v69_full_original_after', 'v69_approved_v3_full_method_after']:
    ranks = []
    for rank in (0, 1):
        if name == 'old40':
            path = Path(next(r for r in native40['ranks'] if r['rank'] == rank)['files']['model'])
            raw = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            raw = {k:v for k,v in raw.items() if '.lora_' in k}
            chain_path = OUT/f'native-full-policy-direction-exchange-v86/iterationfull40_native-rank{rank}-chain.json'
            wanted = json.loads(chain_path.read_bytes())['trainable_SHA256_before']
        else:
            case_name = 'old80_before' if name == 'old80' else name
            case = next(c for c in manifest['cases'] if c['name'] == case_name)
            artifact = case['diagnostic_parameter_partition']['rank_artifacts'][str(rank)]
            path = Path(artifact['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['SHA256']
            raw = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            wanted = artifact['expected_result_trainable_SHA256']
        assert len(raw) == 496 and digest(raw) == wanted
        assert all([str(s) for s in value.placements] == ['S(0)'] for value in raw.values())
        ranks.append({k:local(v).double().clone() for k,v in raw.items()})
        report['sources'].append(dict(state=name, rank=rank, path=str(path),
                                      actual_trainable_SHA256=wanted))
    assert list(ranks[0]) == list(ranks[1])
    modules = {}
    for key in ranks[0]:
        if '.lora_A.' not in key:
            continue
        base = key.split('.lora_A.')[0]
        bkey = key.replace('.lora_A.', '.lora_B.')
        A = torch.cat([r[key] for r in ranks], dim=0)
        Bs = tuple(r[bkey] for r in ranks)
        assert A.shape[0] == 8 and all(B.shape[1] == 8 for B in Bs)
        modules[base] = (A, Bs)
    assert len(modules) == 248
    states[name] = modules
    save()


def difference(left, right):
    """Low-rank factors of B_right A_right - B_left A_left; no dense merge."""
    A0, B0 = left
    A1, B1 = right
    C = tuple(torch.cat([b1-b0, b1], dim=1) for b0,b1 in zip(B0,B1))
    D = torch.cat([A0, A1-A0], dim=0)
    return D, C


def inner(left, right):
    A, Bs = left
    C, Ds = right
    small = sum(B.T@D for B,D in zip(Bs,Ds))
    return scale**2*float(torch.trace(small@(C@A.T)))


def labels(name):
    layer = int(re.search(r'\.layers\.(\d+)\.', name)[1])
    return ['all', 'layers0_7' if layer < 8 else 'layers8_23' if layer < 24 else 'layers24_31',
            f'layer{layer}', name]


comparisons = [(name, 'old80', name) for name in states if name not in ('old40','old80')]
comparisons.extend([
    ('v47_original_minus_same_reward_ablated', 'v47_without_constant_positive_component_after', 'v47_full_original_after'),
    ('v69_original_minus_approved_v3', 'v69_approved_v3_full_method_after', 'v69_full_original_after'),
])
for title, left, right in comparisons:
    groups = defaultdict(lambda:dict(update_energy=0., historical_drift_energy=0., dot=0.))
    for module in states['old80']:
        reference = difference(states['old40'][module], states['old80'][module])
        update = difference(states[left][module], states[right][module])
        uu, hh, uh = inner(update,update), inner(reference,reference), inner(update,reference)
        assert uu >= 0 and hh > 0
        for label in labels(module):
            g = groups[label]
            g['update_energy'] += uu
            g['historical_drift_energy'] += hh
            g['dot'] += uh
    for g in groups.values():
        g.update(effective_update_norm=math.sqrt(g['update_energy']),
                 effective_historical_drift_norm=math.sqrt(g['historical_drift_energy']),
                 cosine_to_effective_historical_drift=g['dot']/math.sqrt(g['update_energy']*g['historical_drift_energy']) if g['update_energy'] else None,
                 projection_fraction_of_effective_historical_drift=g['dot']/g['historical_drift_energy'])
    report['statistics'].append(dict(contrast=title, left=left, right=right,
                                     effective_weight_Frobenius_statistics=dict(groups)))
    save()
report.update(phase='complete_CPU_effective_adapter_update_geometry',
    identity='alpha/r*(B1*A1-B0*A0)=alpha/r*[B1-B0,B1]*[A0;A1-A0]; matrix inner products use small Gram products. These are static saved-weight reporting identities, not model/loss/update implementations.',
    gauge_scope='Effective products and their differences are invariant to equivalent invertible LoRA basis changes; raw A/B displacement is not used as a causal attribution.',
    transpose_scope='The native linear LoRA delta uses PEFT transpose according to fan_in_fan_out. A common transpose preserves all Frobenius inner products reported here.',
    limits=['Effective weight drift is not an input-conditioned Jacobian or the gradient propagation gain; no cause follows from its norm alone.',
            'Saved four-update interventions are finite native outcomes, not reconstructed historical training or independent optimizer component contributions.',
            'Positive/negative alignment with actual40-to80 effective drift is descriptive evidence, not proof that the component caused all long-run gain.',
            'FP64 Gram arithmetic is approximate; no claim of mathematically exact empirical numbers, new tolerance, or accepted root repair.'],
    cuda_initialized=torch.cuda.is_initialized())
assert not torch.cuda.is_initialized()
save()
print(json.dumps(dict(receipt=str(dest), report=report)))
