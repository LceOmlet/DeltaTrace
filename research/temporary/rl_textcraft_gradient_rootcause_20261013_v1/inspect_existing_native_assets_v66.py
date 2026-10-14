"""Read existing historical inputs/checkpoint inventory and official LoRA code.

CPU only. No model construction, data reconstruction, rollout, DT or update.
File inventory does not verify checkpoint contents or recreate a native batch.
"""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT / 'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch = json.loads((RUN / 'launch.json').read_bytes())
env = launch['effective_environment']
sys.path[:0] = env['PYTHONPATH'].split(':')
os.environ.update(env)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from peft import LoraConfig
from peft.tuners.lora.layer import Linear
import torch

path = Path(inspect.getsourcefile(Linear))
forward, start = inspect.getsourcelines(Linear.forward)
defaults = inspect.signature(LoraConfig)
report = dict(unix=time.time(), scope=__doc__, literal_python=sys.executable, prefix=sys.prefix,
              official_PEFT=dict(version=importlib.metadata.version('peft'), path=str(path),
                  file_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  forward_first_line=start, forward_source=''.join(forward),
                  relevant_config_defaults={k: str(defaults.parameters[k].default)
                      for k in ('lora_dropout', 'use_rslora', 'use_dora')}),
              old_rollout_schema={}, checkpoint_inventory={}, capture_inventory=[],
              serialized_files_outside_checkpoints=[], model_calls=0, DT_calls=0,
              backward_calls=0, optimizer_calls=0, rootcause_repair_verified=False)
for step in (40, 41, 60, 61, 80, 81):
    path = RUN / f'textcraft-dt/rollouts/{step}.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        report['old_rollout_schema'][str(step)] = dict(
            path=str(path), bytes=path.stat().st_size, SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=len(rows), keys=sorted(set().union(*(row.keys() for row in rows))),
        )
shared = Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')
for step in (40, 60, 80):
    for label, root in (('primary', RUN), ('shared_existing_copy', shared / str(RUN).lstrip('/'))):
        directory = root / f'textcraft-dt/checkpoints/global_step_{step}'
        expected = [directory / 'data.pt'] + [directory / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
            for rank in (0, 1) for kind in ('model', 'optim', 'extra_state')]
        report['checkpoint_inventory'][f'{label}_{step}'] = [
            dict(path=str(p), exists=p.exists(), bytes=p.stat().st_size if p.exists() else None) for p in expected]
for path in sorted(OUT.rglob('rank*-capture.json')):
    value = json.loads(path.read_bytes())
    if value.get('input'):
        report['capture_inventory'].append(dict(
            path=str(path), completed_step_before_update=value.get('completed_step_before_update'),
            checkpoint_step=value.get('checkpoint_step'), method=value.get('method'), input=value['input'],
            input_SHA256=value.get('input_sha256'), status=value.get('status')))
for root, dirs, files in os.walk(RUN, followlinks=False):
    dirs[:] = [d for d in dirs if d not in ('checkpoints', 'credit-records', '__pycache__', '.git')]
    for name in files:
        if name.endswith(('.pkl', '.pt', '.safetensors', '.parquet')):
            path = Path(root) / name
            report['serialized_files_outside_checkpoints'].append(dict(path=str(path), bytes=path.stat().st_size))
worker_path = Path(env['VERL_ROOT']) / 'verl/workers/fsdp_workers.py'
lines = worker_path.read_text().splitlines()
report['native_LoRA_config_source'] = dict(path=str(worker_path),
    SHA256=hashlib.sha256(worker_path.read_bytes()).hexdigest(),
    excerpts=[dict(first_line=max(1, i-3), lines=lines[max(0, i-4):i+12])
              for i, line in enumerate(lines) if 'LoraConfig(' in line])
assert not torch.cuda.is_initialized()
path = OUT / 'existing-native-inputs-checkpoints-and-LoRA-owner-v66.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
