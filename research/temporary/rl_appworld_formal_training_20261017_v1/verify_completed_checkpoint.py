"""Read only completed native checkpoint artifacts; never save or restore training."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'
method = sys.argv[1]
assert method in ('dt', 'ppo')
job, = [j for j in json.loads((ROOT / 'formal-training.json').read_bytes())['jobs']
        if j.get('task') == 'AppWorld' and j.get('method') == method]
source = json.loads(Path(job['source_receipt']).read_bytes())
directory = Path(job['checkpoints'])
marker = directory / 'latest_checkpointed_iteration.txt'
if not marker.is_file():
    print(json.dumps(dict(method=method, version=job['version'], status='pending_native_completed_checkpoint_marker',
                          checkpoint_path=str(directory), checkpoint_create_calls=0)))
    raise SystemExit(0)
if '--native' not in sys.argv:
    env = dict(os.environ)
    env.update(source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], '-X', 'utf8', __file__, method, '--native'], env=env))

import torch
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

step = int(marker.read_text().strip())
checkpoint = directory / f'global_step_{step}'
assert step > 0
owner = Path(inspect.getsourcefile(FSDPCheckpointManager))
owner_sha = hashlib.sha256(owner.read_bytes()).hexdigest()
assert owner_sha == 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'
receipt = AUDIT / f'{job["version"]}-checkpoint-{step}-verified.json'
if receipt.exists():
    print(json.dumps(dict(status='already_verified_native_checkpoint_metadata', receipt=str(receipt))))
    raise SystemExit(0)

# Same owner RNG contract inspection used by the existing native checkpoint
# reader. No backend state is restored by this verification process.
rng_source = inspect.getsource(FSDPCheckpointManager.load_rng_state)
tree = ast.parse(textwrap.dedent(rng_source))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
argument = function.args.args[-1].arg
required_rng = {n.slice.value for n in ast.walk(function) if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == argument
                and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
required_rng -= {'npu'}
assert 'cuda' in required_rng
roles = []
for role in (('actor', 'critic') if method == 'ppo' else ('actor',)):
    for rank in (0, 1):
        files = {kind: checkpoint / role / f'{kind}_world_size_2_rank_{rank}.pt'
                 for kind in ('model', 'optim', 'extra_state')}
        assert all(p.is_file() and p.stat().st_size > 0 for p in files.values())
        extra = torch.load(files['extra_state'], map_location='cpu', weights_only=False)
        assert extra['lr_scheduler']['last_epoch'] == step
        assert required_rng.issubset(extra['rng'])
        if role == 'actor':
            assert extra.get('rollout_gen_random_states') is not None
        # Official torch mmap reads the native artifacts without retaining a
        # second full copy of the model or optimizer in host RAM.
        model = torch.load(files['model'], map_location='cpu', weights_only=False, mmap=True)
        optim = torch.load(files['optim'], map_location='cpu', weights_only=False, mmap=True)
        assert model and optim['state']
        optimizer_steps = set()
        for value in optim['state'].values():
            value = value['step']
            value = value.to_local() if hasattr(value, 'to_local') else value
            optimizer_steps.add(float(value))
        assert optimizer_steps and min(optimizer_steps) > 0
        roles.append(dict(role=role, rank=rank, scheduler_epoch=step, optimizer_steps=sorted(optimizer_steps),
                          model_entries=len(model), optimizer_entries=len(optim['state']),
                          rng_keys=sorted(extra['rng']), rollout_rng_present=extra.get('rollout_gen_random_states') is not None,
                          files={k:dict(path=str(p),bytes=p.stat().st_size) for k,p in files.items()}))
        del model, optim, extra
data_path = checkpoint / 'data.pt'
data = torch.load(data_path, map_location='cpu', weights_only=False)
assert data
assert not torch.cuda.is_initialized()
result = dict(status='passed_native_shared_checkpoint_state_metadata', collected=time.time(),
              method=method, version=job['version'], step=step, checkpoint=str(checkpoint), marker=str(marker),
              current_marker_after_read=marker.read_text().strip(), roles=roles,
              data_state_keys=list(data), data_SHA256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
              native_RNG_required_keys=sorted(required_rng), owner_path=str(owner), owner_SHA256=owner_sha,
              model_calls=0, optimizer_updates=0, checkpoint_create_calls=0, checkpoint_restore_calls=0,
              scope='Completed native marker and readable saved model/optimizer/scheduler/RNG/data; no training restore or trajectory equivalence claim')
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],receipt=str(receipt),step=step,roles=roles)))
