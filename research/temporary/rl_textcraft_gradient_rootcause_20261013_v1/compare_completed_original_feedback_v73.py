"""Compare two completed original-update diagnostics from saved state, CPU only.

This reads actual parameter artifacts, native metrics and owner configuration.
It does not execute or reimplement any model, loss, backward or optimizer.
"""
from collections import defaultdict
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time

from omegaconf import OmegaConf
import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
directories = [OUT / 'native-common-update-feedback-v47',
               OUT / 'native-full-current-v3-feedback-v69']
states = [json.loads((p / 'result.json').read_bytes()) for p in directories]
manifests = [json.loads((p / 'case-manifest.json').read_bytes()) for p in directories]
observations = [next(o for o in s['observations'] if o['arm'] == 'full_original') for s in states]
assert all(s['phase'].startswith('complete_') for s in states)
assert manifests[0]['original_launch'] == manifests[1]['original_launch']
launch = json.loads(Path(manifests[0]['original_launch']).read_bytes())
environments = [launch['effective_environment'], manifests[1]['worker_effective_environment']]
configs = []
for env in environments:
    cfg = OmegaConf.load(Path(env['VERL_ROOT']) / 'verl/trainer/config/ppo_trainer.yaml')
    for key, value in launch['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 330
    configs.append(OmegaConf.to_container(cfg, resolve=True))


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted(a.keys() | b.keys())
                for item in differences(a.get(key), b.get(key), prefix + '.' + key)]
    return [] if a == b else [dict(field=prefix.lstrip('.'), old=a, new=b)]


report = dict(unix=time.time(), scope=__doc__, source_results={
    str(p / 'result.json'): hashlib.sha256((p / 'result.json').read_bytes()).hexdigest()
    for p in directories}, effective_configuration_differences=differences(*configs),
    source_files=[], native_first_update_metrics={}, parameter_displacement={},
    before_parameter_elements_identical=True, model_calls=0, backward_calls=0,
    optimizer_calls=0, rootcause_repair_verified=False,
    limits=[
        'These are two completed diagnostics, not a repeated model experiment.',
        'A forward metric equality is not an equality check of full activations or gradients.',
        'Parameter differences do not by themselves identify a kernel, routing or numerical cause.',
        'Native four-step displacement is measured directly; it is not reconstructed from losses or gradient norms.',
    ])
for relative in ('verl/workers/actor/dp_actor.py', 'verl/utils/fsdp_utils.py',
                 'verl/utils/torch_functional.py', 'verl/protocol.py',
                 'verl/trainer/config/ppo_trainer.yaml'):
    rows = []
    for env in environments:
        p = Path(env['VERL_ROOT']) / relative
        rows.append(dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
    report['source_files'].append(dict(relative_path=relative, versions=rows,
                                      identical=rows[0]['SHA256'] == rows[1]['SHA256']))
actor_path = Path(environments[1]['VERL_ROOT']) / 'verl/workers/actor/dp_actor.py'
actor_text = actor_path.read_text()
actor_ast = ast.parse(actor_text)
report['native_optimizer_step_source'] = next(ast.get_source_segment(actor_text, n)
    for n in ast.walk(actor_ast) if isinstance(n, ast.FunctionDef) and n.name == '_optimizer_step')
report['native_DataProto_concat_source'] = inspect.getsource(DataProto.concat)
for metric in ('actor/pg_loss', 'actor/ppo_kl', 'actor/pg_clipfrac', 'actor/pg_clipfrac_lower'):
    a, b = [o['original_native_update_metrics'][metric] for o in observations]
    assert len(a) == len(b) == 32
    report['native_first_update_metrics'][metric] = dict(
        first_eight_identical=a[:8] == b[:8],
        first_eight_old=a[:8], first_eight_new=b[:8],
        first_difference_index=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
report['native_preclip_grad_norms'] = [o['original_native_update_metrics']['actor/grad_norm'] for o in observations]

totals = defaultdict(float)
partitions = defaultdict(lambda: defaultdict(float))
dtype_sets = [set(), set()]
snapshot_sources = []
for rank in (0, 1):
    values = []
    for side, observation in enumerate(observations):
        pair = []
        for when in ('before', 'after'):
            item = next(r for r in observation[when] if r['rank'] == rank)
            p = Path(item['trainable_snapshot'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['snapshot_SHA256']
            pair.append(torch.load(p, map_location='cpu', weights_only=False))
            snapshot_sources.append(dict(path=str(p), SHA256=item['snapshot_SHA256']))
            dtype_sets[side].update(str(v.dtype) for v in pair[-1].values())
        values.append(pair)
    old_before, old_after = values[0]
    new_before, new_after = values[1]
    assert old_before.keys() == old_after.keys() == new_before.keys() == new_after.keys()
    for key in old_before:
        assert torch.equal(old_before[key], new_before[key]), (rank, key)
        a = old_after[key].double() - old_before[key].double()
        b = new_after[key].double() - new_before[key].double()
        layer = key.split('.layers.', 1)[1].split('.', 1)[0]
        factor = 'A' if '.lora_A.' in key else 'B'
        ss = dict(old_ss=float(a.square().sum()), new_ss=float(b.square().sum()),
                  difference_ss=float((b-a).square().sum()), dot=float((a*b).sum()),
                  elements=a.numel(), unequal_after_elements=int((old_after[key] != new_after[key]).sum()))
        for name, value in ss.items():
            totals[name] += value
            partitions['layer_' + layer][name] += value
            partitions['LoRA_' + factor][name] += value


def geometry(values):
    return dict(old_norm=math.sqrt(values['old_ss']), new_norm=math.sqrt(values['new_ss']),
                difference_norm=math.sqrt(values['difference_ss']),
                cosine=values['dot'] / math.sqrt(values['old_ss'] * values['new_ss']),
                difference_over_old=math.sqrt(values['difference_ss'] / values['old_ss']),
                elements=int(values['elements']), unequal_after_elements=int(values['unequal_after_elements']))


report['parameter_displacement'] = dict(total=geometry(totals),
    partitions={k: geometry(v) for k, v in partitions.items()},
    actual_snapshot_dtypes=[sorted(v) for v in dtype_sets], source_snapshots=snapshot_sources)
assert not torch.cuda.is_initialized()
dest = OUT / 'completed-original-feedback-comparison-v73.json'
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))
