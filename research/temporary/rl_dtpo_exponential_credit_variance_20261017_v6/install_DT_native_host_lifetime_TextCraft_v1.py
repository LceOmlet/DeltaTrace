"""Bind the tested DT resource-only function at an original worker RPC boundary.

Keep the resident actor, optimizer, sampler and numerical owners. The original
module remains the numerical baseline; the completed runtime override records
the candidate function's actual source separately. No training RPC is added.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
RESOURCE = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3'
VERSION = 'DT-native-host-lifetime-TextCraft-20261017-v1'
ORIGINAL_SHA = '198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd'
CANDIDATE_SHA = '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'

job, = [j for j in json.loads((ROOT / 'formal-training.json').read_bytes())['jobs']
        if j.get('task') == 'TextCraft' and j.get('method') == 'dt']
source = json.loads(Path(job['source_receipt']).read_bytes())
receipt = Path(job['output']).parent / ('runtime-overrides/' + VERSION + '.json')
commit = sys.argv[sys.argv.index('--patch-commit') + 1]
assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
if receipt.exists():
    print(json.dumps(dict(status='completed_runtime_override_retained', receipt=str(receipt))))
    raise SystemExit(0)

if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    result = subprocess.run([env['VENV_PYTHON'], '-X', 'utf8', __file__,
                             '--patch-commit', commit, '--client'],
                            env=env, capture_output=True, text=True)
    (AUDIT / (VERSION + '-client.stdout')).write_text(result.stdout)
    (AUDIT / (VERSION + '-client.stderr')).write_text(result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, file=sys.stderr, end='')
    raise SystemExit(result.returncode)

import psutil
import ray

driver = psutil.Process(job['pid'])
assert (driver.pid, driver.create_time()) == (3823185, 1792200693.05) == (job['pid'], job['birth'])
workers = [p for p in driver.children(recursive=True) if 'WorkerDict' in p.name()]
assert {p.pid for p in workers} == {3861049, 3863060}
tested = json.loads((RESOURCE / 'execution2/native-comparison-job.json').read_bytes())
assert tested['status'] == 'completed' and tested['native_DT_vectors_equal'] == [True]
prepared = json.loads((RESOURCE / 'prepared.json').read_bytes())
assert prepared['original_SHA256'] == ORIGINAL_SHA and prepared['candidate_SHA256'] == CANDIDATE_SHA
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))


def bind_tested_native_resource_function(worker):
    import ast
    import copy
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import psutil
    import torch

    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    resource = root / 'receipts/dt-native-replay-host-lifetime-20261017-v3'
    facts = json.loads((resource / 'prepared.json').read_bytes())
    original_path = Path(facts['original_root']) / facts['changed_sources'][0]
    candidate_path = Path(facts['candidate_root']) / facts['changed_sources'][0]
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == facts['original_SHA256']
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == facts['candidate_SHA256']
    process = psutil.Process()
    assert process.create_time() == {3861049: 1792201006.87, 3863060: 1792201026.62}[process.pid]
    modules = {id(m): m for m in list(sys.modules.values()) if m is not None
               and getattr(m, '__file__', None)
               and Path(m.__file__).resolve() == original_path.resolve()}
    module, = modules.values()
    tree = ast.parse(original_path.read_bytes())
    cls, = [c for c in tree.body if isinstance(c, ast.ClassDef)
            and any(isinstance(f, ast.FunctionDef) and f.name == 'attribute' for f in c.body)]
    raw = getattr(getattr(module, cls.name), 'attribute')

    def owned_code(path):
        parsed = ast.parse(path.read_bytes())
        c, = [c for c in parsed.body if isinstance(c, ast.ClassDef) and c.name == cls.name]
        fn, = [f for f in c.body if isinstance(f, ast.FunctionDef) and f.name == 'attribute']
        fn = copy.deepcopy(fn)
        fn.decorator_list = []
        namespace = dict(raw.__globals__)
        namespace['os'] = os
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec', dont_inherit=True), namespace)
        return namespace['attribute'].__code__

    baseline = owned_code(original_path)
    candidate = owned_code(candidate_path)
    assert raw.__code__.co_code in (baseline.co_code, candidate.co_code)
    assert raw.__code__.co_freevars == candidate.co_freevars == ()
    assert raw.__code__.co_argcount == candidate.co_argcount
    for frame in sys._current_frames().values():
        while frame is not None:
            assert frame.f_code is not raw.__code__, 'Bind only after the original attribute call returned'
            frame = frame.f_back
    actor, = [v for v in worker.worker_dict.values() if getattr(v, 'actor', None) is not None]
    before = (actor.actor_lr_scheduler.last_epoch, len(actor.actor_optimizer.state))
    old_code = hashlib.sha256(raw.__code__.co_code).hexdigest()
    raw.__globals__['os'] = os
    raw.__code__ = candidate.replace(co_qualname=raw.__code__.co_qualname)
    os.environ['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] = '1'
    after = (actor.actor_lr_scheduler.last_epoch, len(actor.actor_optimizer.state))
    assert before == after and raw.__code__.co_code == candidate.co_code
    assert Path(raw.__code__.co_filename) == candidate_path
    # Statistics only; reset after the old RPC, before measuring a future one.
    torch.cuda.memory.reset_peak_host_memory_stats()
    return dict(pid=process.pid, birth=process.create_time(),
                module_source=str(original_path), module_source_SHA256=facts['original_SHA256'],
                actual_function_source=raw.__code__.co_filename,
                actual_function_source_SHA256=facts['candidate_SHA256'],
                previous_live_code_SHA256=old_code,
                live_code_SHA256=hashlib.sha256(raw.__code__.co_code).hexdigest(),
                environment_runtime_override={'DT_NATIVE_HOST_CACHE_REPLAY_LAYER': '1'},
                actor_scheduler_epoch=after[0], actor_optimizer_entries=after[1],
                native_host_reserved_current=torch.cuda.memory.host_memory_stats()['reserved_bytes.current'],
                native_host_peak_counter_reset=True,
                model_calls=0, DT_calls=0, optimizer_updates=0, tensor_state_writes=0)


request = AUDIT / (VERSION + '-job.json')
request.write_text(json.dumps(dict(status='queued_resource_binding_on_original_worker_RPC_boundary',
                                  pid=os.getpid(), birth=psutil.Process().create_time(),
                                  driver_pid=driver.pid, driver_birth=driver.create_time(),
                                  collected=time.time(), local_patch_commit=commit), indent=2) + '\n')
ray.init(address=host + ':' + port, log_to_driver=False)
try:
    named = [a for a in ray.util.list_named_actors(all_namespaces=True)
             if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(named) == 2
    handles = [ray.get_actor(a['name'], namespace=a['namespace']) for a in named]
    applied = ray.get([a.execute_with_func_generator.remote(func=bind_tested_native_resource_function) for a in handles])
    assert {x['pid'] for x in applied} == {3861049, 3863060}
    assert len({x['live_code_SHA256'] for x in applied}) == 1
    record = dict(status='completed_tested_resource_function_binding', collected=time.time(),
                  task='TextCraft', method='dt', method_version='v6', resource_version=VERSION,
                  upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
                  local_patch_commit=commit, actual_driver=dict(pid=driver.pid, birth=driver.create_time()),
                  original_source_receipt=job['source_receipt'],
                  source_bindings={str(Path(prepared['candidate_root']) / prepared['changed_sources'][0]): CANDIDATE_SHA},
                  actual_function_overrides=applied,
                  environment_runtime_override={'DT_NATIVE_HOST_CACHE_REPLAY_LAYER': '1'},
                  original_module_and_DT_ROOT_retained=True,
                  verification=str(RESOURCE / 'execution2/native-comparison-summary-v1.json'),
                  formal_training_restarts=0, model_calls=0, DT_calls=0, optimizer_updates=0,
                  first_future_natural_resource_execution_observed=False,
                  full_two_rank_capacity_verified=False, four_resident_capacity_verified=False)
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    for name in ('formal-training.json', 'active-training.json', 'active-source.json',
                 'receipts/appworld-formal-training-20261017-v1/formal-training.json'):
        path = ROOT / name
        manifest = json.loads(path.read_bytes())
        for row in manifest['jobs']:
            if row.get('task') == 'TextCraft' and row.get('method') == 'dt':
                assert (row['pid'], row['birth']) == (driver.pid, driver.create_time())
                row.setdefault('completed_runtime_overrides', []).append(dict(
                    status=record['status'], receipt=str(receipt), resource_version=VERSION,
                    local_patch_commit=commit, source_bindings=record['source_bindings'],
                    actual_function_overrides=applied,
                    environment_runtime_override=record['environment_runtime_override']))
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    request.write_text(json.dumps(dict(status=record['status'], receipt=str(receipt),
                                      completed=time.time()), indent=2) + '\n')
    print(json.dumps(record))
finally:
    ray.shutdown()
