"""Compose the existing bounded vLLM profiler with the current eight-rank client.

The original ProfilerConfig, TorchProfilerWrapper and generate wrapper own all
profiling behavior. This adapter supplies only current identities and output
paths. It does not implement a scheduler, generation loop or training logger.
"""
import ast
import hashlib
import inspect
import os
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-register-threads-20261017-v1'
out = AUDIT / 'native-rollout-profile-one-batch-v1'
out.mkdir(exist_ok=True)

profile_source = AUDIT / 'source/original_profile_formal_vllm_owner.py'
assert hashlib.sha256(profile_source.read_bytes()).hexdigest() == 'cfde2c02dd1ef8d494ed77e9e96672526b023a884867a43172dcff3f25cb24f0'
tree = ast.parse(profile_source.read_text())
remote_call = next(n for n in tree.body if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call)
                   and getattr(n.value.func, 'id', '') == 'remote')
literal = next(n.value for n in ast.walk(remote_call)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and 'def begin_native(' in n.value)
body = 'def begin_native(' + literal.split('def begin_native(', 1)[1]
body = body.split('\nray.init(', 1)[0]
functions = [n for n in ast.parse(body).body if isinstance(n, ast.FunctionDef)
             and n.name in ('begin_native', 'finish_native', 'install')]
assert {n.name for n in functions} == {'begin_native', 'finish_native', 'install'}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(profile_source), 'exec'), globals())


def read(worker):
    """Check the actual original worker profiler slot, then call its old adapter."""
    import psutil
    from pathlib import Path

    expected = {r['pid']: r['birth'] for r in json.loads((RUN /
        'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    engine = owner.rollout.inference_engine

    def preflight(wrapper):
        import hashlib
        import inspect
        import os
        from pathlib import Path
        import torch

        native = wrapper.worker
        path = Path(inspect.getfile(type(native.model_runner)))
        return dict(pid=os.getpid(), profiler_is_none=native.profiler is None,
            supported_activities=sorted(str(x) for x in torch.profiler.supported_activities()),
            runner_source=str(path), runner_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())

    pre = engine.collective_rpc(preflight)
    assert len(pre) == 1 and all(row['profiler_is_none'] for row in pre)
    assert all(set(row['supported_activities']) == {'ProfilerActivity.CPU', 'ProfilerActivity.CUDA'}
               for row in pre)
    assert all(row['runner_SHA256'] == '6ed94945506fe0baa62e1f729aae824d857e93e4980780fec60c35440016e1ca'
               for row in pre)
    result = install(worker)
    result.update(birth=psutil.Process().create_time(), preflight=pre,
                  scheduler_epoch=owner.actor_lr_scheduler.last_epoch,
                  scope='One existing generate batch; native profiler maximum16 worker iterations')
    return result


client_source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/read_native_generation_buffers.py'
assert hashlib.sha256(client_source.read_bytes()).hexdigest() == '23659d158380ff8d505c13164f5c5e1ba4f308a44ba3519c7d0eb756d7221685'
client = client_source.read_text().replace('receipts/six-run-native-utilization-20261017-v1',
    'receipts/six-run-fresh-after-register-threads-20261017-v1')
client = client.replace('runs/textcraft-dtpo-v6-8gpu-200-20261017-v1',
    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3')
client = client.replace('native-generation-position-buffers-20261017-v1',
    'bounded-native-generation-profile-20261018-v1')
client = client.replace('readonly_native_metadata_pending_original_worker_boundary',
    'bounded_native_profiler_pending_original_worker_boundary')
client = client.replace('read_existing_native_generation_buffer_metadata',
    'original_bounded_native_profiler_installed_pending_natural_batch')
client = client.replace('additional_model_tensor_copy_optimizer_sync_or_config_calls=0',
    'additional_generation_or_optimizer_calls=0')
nodes = [n for n in ast.parse(client).body
         if not (isinstance(n, ast.FunctionDef) and n.name == 'read')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(client_source), 'exec'), globals())
