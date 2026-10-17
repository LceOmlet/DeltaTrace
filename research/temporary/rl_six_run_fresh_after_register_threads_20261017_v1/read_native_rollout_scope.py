"""Read the original worker's rollout-context state; install no hooks."""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/read_native_generation_buffers.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '23659d158380ff8d505c13164f5c5e1ba4f308a44ba3519c7d0eb756d7221685'


def read(worker):
    import hashlib
    import inspect
    import os
    from pathlib import Path
    import psutil

    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    manager = owner.rollout_sharding_manager
    paths = (inspect.getfile(type(owner)), inspect.getfile(type(manager)))
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
        owner_rollout_context_open=getattr(owner, '_owner_rollout_context_open', None),
        base_sync_done=getattr(manager, 'base_sync_done', None),
        scheduler_epoch=owner.actor_lr_scheduler.last_epoch,
        actual_owner_sources={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        installed_hooks=0, additional_model_tensor_copy_optimizer_sync_or_config_calls=0)


code = source.read_text().replace('receipts/six-run-native-utilization-20261017-v1',
    'receipts/six-run-fresh-after-register-threads-20261017-v1')
code = code.replace('runs/textcraft-dtpo-v6-8gpu-200-20261017-v1',
    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3')
code = code.replace('native-generation-position-buffers-20261017-v1',
    'native-rollout-context-scope-20261018-v1')
code = code.replace('read_existing_native_generation_buffer_metadata',
    'read_existing_native_rollout_scope_flags')
nodes = [n for n in ast.parse(code).body
         if not (isinstance(n, ast.FunctionDef) and n.name == 'read')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), globals())
