"""Read the pinned native actor/FSDP gradient accumulation contract on CPU.

This checks whether saved differences of native .grad can be interpreted as
per-B4 increments. It neither executes nor substitutes a gradient reducer.
"""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time

import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'


def owner(path, words, radius=3):
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    wanted = set()
    for n, line in enumerate(lines):
        if any(w in line for w in words):
            wanted.update(range(max(0, n-radius), min(len(lines), n+radius+1)))
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                excerpts=[dict(line=n+1, text=lines[n]) for n in sorted(wanted)])


actor_path = Path(inspect.getsourcefile(DataParallelPPOActor))
assert hashlib.sha256(actor_path.read_bytes()).hexdigest() == ACTOR_SHA
actor_ast = ast.parse(actor_path.read_text())
klass = next(n for n in actor_ast.body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
update = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
lines = actor_path.read_text().splitlines()
update_source = '\n'.join(lines[update.lineno-1:update.end_lineno])
modules = {}
for key, name in [
    ('FSDP_module', 'torch.distributed.fsdp._fully_shard._fully_shard'),
    ('FSDP_param_group', 'torch.distributed.fsdp._fully_shard._fsdp_param_group'),
    ('FSDP_state', 'torch.distributed.fsdp._fully_shard._fsdp_state'),
]:
    module = importlib.import_module(name)
    modules[key] = owner(inspect.getsourcefile(module), [
        'reduce_grads', 'all_reduce_grads', 'post_backward', 'set_requires_gradient_sync',
        'set_is_last_backward', 'unsharded_accumulated_grad',
    ])

report = dict(unix=time.time(), scope=__doc__, torch_version=torch.__version__,
    actor=owner(actor_path, ['gradient_accumulation', 'loss.backward()', 'zero_grad',
                            'no_sync', 'set_requires_gradient_sync', 'set_is_last_backward']),
    actor_update_lines=[update.lineno, update.end_lineno],
    update_has_no_sync='no_sync' in update_source,
    update_has_requires_gradient_sync='set_requires_gradient_sync' in update_source,
    update_has_set_is_last_backward='set_is_last_backward' in update_source,
    worker=owner(inspect.getsourcefile(ActorRolloutRefWorker),
                 ['set_requires_gradient_sync', 'set_is_last_backward', 'no_sync']),
    fsdp=modules, model_calls=0, backward_calls=0, optimizer_calls=0,
    cuda_initialized=torch.cuda.is_initialized(),
    limits=[
        'Source defaults are not a saved readout of every diagnostic runtime flag.',
        'Saved Gram matrices are differences of observed accumulated FP32 shards, not independent-token covariance or Fisher/NTK.',
        'No training cause or repair is accepted by this read-only contract audit.',
    ])
assert not report['cuda_initialized']
dest = OUT / 'native-accumulation-contract-v107.json'
dest.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest), report=report)))
