"""Prepare the existing authorized PPO pause/full-restore workflow, CPU only."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT / 'native-PPO-control-v9-job.json').read_bytes())
RUN = Path(job['run'])
launch = json.loads((RUN / 'launch.json').read_bytes())
env = launch['effective_environment']
sys.path[:0] = env['PYTHONPATH'].split(':')
os.environ.update(env)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

p = psutil.Process(job['pid'])
assert p.create_time() == job['birth'] and p.status() != psutil.STATUS_ZOMBIE
directory = Path(launch['options']['trainer.default_local_dir'])
marker = directory / 'latest_checkpointed_iteration.txt'
step = int(marker.read_text().strip())
assert step >= 20, 'Keep PPO running until its next existing native full checkpoint is complete.'
checkpoint = directory / f'global_step_{step}'
load_rng_source = inspect.getsource(FSDPCheckpointManager.load_rng_state)
tree = ast.parse(textwrap.dedent(load_rng_source))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
state_argument = function.args.args[-1].arg
required_rng = {n.slice.value for n in ast.walk(function) if isinstance(n, ast.Subscript)
    and isinstance(n.value, ast.Name) and n.value.id == state_argument
    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
assert required_rng, 'Inspect the actual native RNG contract before proceeding.'
# The native function has mutually exclusive CUDA/NPU branches. This C550
# training runtime uses CUDA; demanding the unused NPU state is incorrect.
other_backend_rng = required_rng.intersection({'npu'})
required_rng -= other_backend_rng
assert 'cuda' in required_rng
roles = []
for role in ('actor', 'critic'):
    for rank in (0, 1):
        files = {kind: checkpoint / role / f'{kind}_world_size_2_rank_{rank}.pt'
                 for kind in ('model', 'optim', 'extra_state')}
        assert all(path.is_file() and path.stat().st_size > 0 for path in files.values())
        extra = torch.load(files['extra_state'], map_location='cpu', weights_only=False)
        assert extra['lr_scheduler']['last_epoch'] == step
        assert required_rng.issubset(extra['rng'])
        if role == 'actor':
            assert extra.get('rollout_gen_random_states') is not None
        optim = torch.load(files['optim'], map_location='cpu', weights_only=False)
        states = optim['state']
        steps = set()
        for state in states.values():
            value = state['step']
            value = value.to_local() if hasattr(value, 'to_local') else value
            steps.add(float(value))
        assert states and steps == {float(4 * step)}, (role, rank, steps)
        roles.append(dict(role=role, rank=rank, scheduler=step, optimizer_steps=sorted(steps),
            optimizer_entries=len(states), rng_keys=sorted(extra['rng']),
            rollout_rng_saved=role != 'actor' or extra.get('rollout_gen_random_states') is not None,
            files={kind: dict(path=str(path), bytes=path.stat().st_size) for kind, path in files.items()}))
        del optim, extra
data_path = checkpoint / 'data.pt'
data_state = torch.load(data_path, map_location='cpu', weights_only=False)
assert data_state
protected = []
dt_job = json.loads((OUT / 'DTPO-variance-v3-job.json').read_bytes())
grpo = next(j for j in json.loads((ROOT / 'formal-training.json').read_bytes())['jobs']
            if j.get('task') == 'TextCraft' and j.get('method') == 'grpo')
for role, identity in (('DTPO_v3', dt_job), ('GRPO', grpo)):
    q = psutil.Process(identity['pid'])
    assert q.create_time() == identity['birth'] and q.status() != psutil.STATUS_ZOMBIE
    protected.append(dict(role=role, pid=q.pid, birth=q.create_time(), required_alive=True))
physical = subprocess.check_output(['mx-smi'], text=True, timeout=30)
for m in re.finditer(r'\|\s*([67])\s+(\d+)\s+', physical.split('| Process:')[-1]):
    q = psutil.Process(int(m[2]))
    protected.append(dict(role='other_project_GPU'+m[1], pid=q.pid, birth=q.create_time(), required_alive=False))
owner = Path(inspect.getsourcefile(FSDPCheckpointManager))
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'
assert marker.read_text().strip() == str(step)
assert not torch.cuda.is_initialized()
report = dict(unix=time.time(), status='planned_only_PPO_still_running', job=job,
    completed_iteration=step, checkpoint=str(checkpoint), marker=str(marker),
    data_SHA256=hashlib.sha256(data_path.read_bytes()).hexdigest(), native_data_state_keys=list(data_state),
    role_checkpoint_states=roles, native_RNG_required_keys=sorted(required_rng),
    native_RNG_inapplicable_backend_keys=sorted(other_backend_rng),
    native_RNG_restore_owner=dict(path=str(owner), SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(), source=load_rng_source),
    protected_jobs=protected, physical_mx_smi=physical,
    instruction='Reuse the user-authorized temporary PPO pause for a finite gradient diagnostic, then restore the same original run through native full-checkpoint loading.',
    source_hyperparameters_changed=False, process_signals_sent=0, new_checkpoints_created=False)
path = OUT / 'PPO-native-pause-for-v69-plan-v70.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(path), report=report)))
