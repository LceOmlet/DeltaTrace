"""Read A/B balance of stored native Adam moments; never step an optimizer.

The existing history shows that A coordinates moved appreciably despite the
small A share of an actual late PG. Inspect the moment ratio at the same saved
point, including a saved control state, before attributing this to the method.
"""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import psutil
import torch

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
history_path = OUT/'temporal-adapter-states-v2.json'
history = json.loads(history_path.read_bytes())
control_path = OUT/'native-full-policy-saved-update-feedback-v90/case-manifest.json'
control_manifest = json.loads(control_path.read_bytes())
control = next(c for c in control_manifest['cases'] if c['name'] == 'GRPO160_same_full80_seed')
contract_path = OUT/'native-optimizer-constructor-source-v110.json'
contract = json.loads(contract_path.read_bytes())
assert contract['worker']['SHA256'] == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
assert any('actor_module_fsdp.parameters()' in x['text'] for x in contract['worker']['lines'])
assert any('self.optimizer.state_dict()' in x['text'] for x in contract['checkpoint_manager']['lines'])
report = dict(unix=time.time(), phase='reading_existing_native_moments', scope=__doc__,
    pid=os.getpid(), birth=psutil.Process().create_time(),
    diagnostic_source=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    source_contract=dict(path=str(contract_path), SHA256=hashlib.sha256(contract_path.read_bytes()).hexdigest()),
    cases=[], model_calls=0, backward_calls=0, optimizer_steps=0, formal_training_changes=0,
    GRPO_training_reruns=0, cuda_initialized=torch.cuda.is_initialized())
dest = OUT/'native-moment-coordinate-balance-v111.json'


def save():
    report['unix'] = time.time()
    dest.write_text(json.dumps(report, indent=2)+'\n')


cases = []
for step in (40, 80):
    c = next(c for c in history['checkpoints'] if c['step'] == step)
    cases.append(dict(name=f'oldDT{step}', step=step,
                      ranks=[dict(rank=r['rank'], names=r['LoRA_tensor_names'],
                                  optimizer=r['files']['optimizer']) for r in c['ranks']]))
cases.append(dict(name='saved_GRPO160', step=160,
    ranks=[dict(rank=r, names=control['diagnostic_parameter_partition']['rank_artifacts'][str(r)]['selected_keys'],
                optimizer=str(Path(control['checkpoint_path'])/f'optim_world_size_2_rank_{r}.pt')) for r in (0, 1)]))
for case in cases:
    aggregate = defaultdict(lambda: dict(first_moment_squared=0., second_moment_sum=0.,
                                         adaptive_increment_squared=0., elements=0))
    files = []
    for rr in case['ranks']:
        path = Path(rr['optimizer'])
        opt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert len(opt['param_groups']) == 1
        group = opt['param_groups'][0]
        ids = [p for p in group['params'] if p in opt['state']]
        assert len(ids) == len(rr['names']) == len(opt['state']) == 496
        beta1, beta2 = group['betas']
        steps = set()
        for name, pid in zip(rr['names'], ids):
            s = opt['state'][pid]
            def local(x):
                return (x.to_local() if hasattr(x, 'to_local') else x).double()
            m, v = local(s['exp_avg']), local(s['exp_avg_sq'])
            count = float(s['step'])
            steps.add(count)
            side = 'A' if '.lora_A.' in name else 'B'
            ratio = group['lr']*(m/(1-beta1**count))/(v.sqrt()/math.sqrt(1-beta2**count)+group['eps'])
            row = aggregate[side]
            row['first_moment_squared'] += float(m.square().sum())
            row['second_moment_sum'] += float(v.sum())
            row['adaptive_increment_squared'] += float(ratio.square().sum())
            row['elements'] += m.numel()
        assert steps == {float(4*case['step'])}
        files.append(dict(rank=rr['rank'], path=str(path), bytes=path.stat().st_size,
                          modified_unix=path.stat().st_mtime, optimizer_step=4*case['step'],
                          native_configuration={k:group[k] for k in ['lr','betas','eps','weight_decay','amsgrad','foreach','fused']}))
        del opt
    totals = {k:math.fsum(v[k] for v in aggregate.values()) for k in
              ['first_moment_squared','second_moment_sum','adaptive_increment_squared','elements']}
    for row in aggregate.values():
        row.update(first_moment_energy_fraction=row['first_moment_squared']/totals['first_moment_squared'],
            second_moment_energy_fraction=row['second_moment_sum']/totals['second_moment_sum'],
            adaptive_increment_energy_fraction=row['adaptive_increment_squared']/totals['adaptive_increment_squared'])
    report['cases'].append(dict(name=case['name'], iteration=case['step'], files=files,
        statistics=dict(aggregate), total_adaptive_increment_norm=math.sqrt(totals['adaptive_increment_squared'])))
    save()
report.update(phase='complete_CPU_native_moment_coordinate_balance',
    scope_qualification='The native bias-corrected moment-ratio statistic is evaluated in FP64; it excludes weight decay/finite-precision CUDA operations. No optimizer call, new gradient, or replacement optimizer is made.',
    v109_constructor_audit_qualification='The v109 literal torch.optim.AdamW source search returned no lines because the actual pinned owner uses the optim.AdamW alias. v110 captures the alias, native actor_module_fsdp.parameters() order and direct native optimizer.state_dict() ownership; v109 raw evidence is preserved.',
    limits=['Adam first/second moments use different filters and include clipped joint actor gradients. No variance/noise decomposition is claimed.',
            'The saved control160 and DT40/80 are different training histories and ages, not a randomized algorithm comparison or native control policy gradient.',
            'A/B update balance alone cannot identify the direction that causes temporal internal gain or accept a root repair.'],
    rootcause_repair_verified=False)
assert not torch.cuda.is_initialized()
save()
print(json.dumps(dict(receipt=str(dest), report=report)))
