"""Patch the existing critic RPC resource tail; preserve its numerical body."""
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
PATCH = AUDIT / 'critic-rpc-host-cache-release-20261017-v2'
EXPECTED = '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
UPSTREAM = '20bd331bdbc9026a5668e11362178e10ab7400c8'
job, = [j for j in json.loads((ROOT / 'formal-training.json').read_bytes())['jobs']
        if j.get('task') == 'TextCraft' and j.get('method') == 'ppo']
source = json.loads(Path(job['source_receipt']).read_bytes())
owner = Path(source['effective_environment']['VERL_ROOT']) / 'verl/workers/fsdp_workers.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def critic_function(tree):
    matches = [(cls, fn) for cls in tree.body if isinstance(cls, ast.ClassDef)
               for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == 'update_critic']
    assert len(matches) == 1
    return matches[0]


if '--prepare' in sys.argv:
    before = owner.read_bytes()
    assert digest(before) == EXPECTED
    text = before.decode()
    original_tree = ast.parse(text)
    cls, fn = critic_function(original_tree)
    assert isinstance(fn.body[-1], ast.Return) and ast.unparse(fn.body[-1]) == 'return output'
    assert ast.unparse(fn.body[-2]) == "output = output.to('cpu')"
    insertion = ('        if os.environ.get("VERL_RELEASE_UNUSED_HOST_CACHE") == "1":\n'
                 '            from verl.utils.native_host_memory import release_unused_host_cache\n'
                 '            release_unused_host_cache("critic_completed_RPC_after_offload")\n')
    lines = text.splitlines(True)
    lines.insert(fn.body[-1].lineno - 1, insertion)
    patched = ''.join(lines).encode()
    patched_tree = ast.parse(patched)
    _, changed = critic_function(patched_tree)
    branch = changed.body.pop(-2)
    assert isinstance(branch, ast.If)
    assert ast.dump(patched_tree, include_attributes=False) == ast.dump(original_tree, include_attributes=False)
    compile(patched, str(owner), 'exec')
    PATCH.mkdir(exist_ok=False)
    (PATCH / 'original-fsdp_workers.py').write_bytes(before)
    (PATCH / 'patched-fsdp_workers.py').write_bytes(patched)
    (PATCH / 'owner.patch').write_text(''.join(difflib.unified_diff(
        text.splitlines(True), patched.decode().splitlines(True),
        fromfile=str(owner), tofile=str(owner), n=0)))
    record = dict(status='prepared_not_deployed', collected=time.time(), owner=str(owner),
                  original_SHA256=EXPECTED, patched_SHA256=digest(patched),
                  original_class=cls.name, original_function='update_critic',
                  upstream_commit=UPSTREAM, original_source_receipt=job['source_receipt'],
                  unchanged_module_AST_after_removing_resource_branch=True,
                  default_disabled_path_unchanged=True, model_calls=0, optimizer_updates=0,
                  scope='One existing-owner call after original CPU offload and output transfer; no PPO body replacement')
    (PATCH / 'prepared.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    raise SystemExit(0)

assert '--deploy' in sys.argv
commit = sys.argv[sys.argv.index('--patch-commit') + 1]
assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
prepared = json.loads((PATCH / 'prepared.json').read_bytes())
assert prepared['owner'] == str(owner)
receipt = Path(job['output']).parent / 'runtime-overrides/native-critic-RPC-host-cache-release-20261017-v2.json'
if receipt.exists():
    print(json.dumps(dict(status='completed_runtime_binding_retained', receipt=str(receipt))))
    raise SystemExit(0)

if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    result = subprocess.run([env['VENV_PYTHON'], '-X', 'utf8', __file__, '--deploy',
                             '--patch-commit', commit, '--client'],
                            env=env, capture_output=True, text=True)
    (PATCH / 'client.stdout').write_text(result.stdout)
    (PATCH / 'client.stderr').write_text(result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)

import psutil
import ray

driver = psutil.Process(job['pid'])
assert driver.pid == 3823147 and driver.create_time() == job['birth'] == 1792200692.49
workers = [p for p in driver.children(recursive=True) if 'WorkerDict' in p.name()]
assert {p.pid for p in workers} == {3874156, 3876616}
assert all(p.name() == 'ray::WorkerDict' for p in workers)
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))


def install_same_owner_resource_tail(worker):
    import ast
    import copy
    import hashlib
    import inspect
    import json
    from pathlib import Path
    import sys
    import psutil
    import verl.utils.native_host_memory as resource_owner

    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    directory = root / 'receipts/dtpo-exponential-credit-variance-20261017-v6/critic-rpc-host-cache-release-20261017-v2'
    facts = json.loads((directory / 'prepared.json').read_bytes())
    path = Path(facts['owner'])
    current = psutil.Process()
    assert current.create_time() == {3874156: 1792201135.09, 3876616: 1792201155.53}[current.pid]
    critic, = [item for item in worker.worker_dict.values() if getattr(item, 'critic', None) is not None]
    raw = inspect.unwrap(critic.update_critic.__func__)
    assert Path(raw.__code__.co_filename) == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() in (facts['original_SHA256'], facts['patched_SHA256'])

    def compile_owned_function(filename):
        tree = ast.parse(filename.read_bytes())
        cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == facts['original_class']]
        fn, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update_critic']
        fn = copy.deepcopy(fn)
        fn.decorator_list = []
        namespace = dict(raw.__globals__)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec', dont_inherit=True), namespace)
        return namespace['update_critic'].__code__

    original = compile_owned_function(directory / 'original-fsdp_workers.py')
    patched = compile_owned_function(directory / 'patched-fsdp_workers.py')
    assert raw.__code__.co_code in (original.co_code, patched.co_code)
    assert raw.__code__.co_freevars == patched.co_freevars == ()
    assert raw.__code__.co_argcount == patched.co_argcount
    assert callable(resource_owner.release_unused_host_cache)
    for frame in sys._current_frames().values():
        while frame is not None:
            assert frame.f_code is not raw.__code__, 'Do not replace an executing critic function'
            frame = frame.f_back
    state_before = (critic.critic_lr_scheduler.last_epoch, len(critic.critic_optimizer.state))
    before_code_SHA256 = hashlib.sha256(raw.__code__.co_code).hexdigest()
    raw.__code__ = patched.replace(co_qualname=raw.__code__.co_qualname)
    assert raw.__code__.co_code == patched.co_code
    state_after = (critic.critic_lr_scheduler.last_epoch, len(critic.critic_optimizer.state))
    assert state_before == state_after
    return dict(pid=current.pid, birth=current.create_time(), owner=str(path),
                before_code_SHA256=before_code_SHA256,
                patched_code_SHA256=hashlib.sha256(raw.__code__.co_code).hexdigest(),
                native_helper_path=resource_owner.__file__,
                native_helper_SHA256=hashlib.sha256(Path(resource_owner.__file__).read_bytes()).hexdigest(),
                critic_scheduler_epoch=state_after[0], critic_optimizer_state_entries=state_after[1],
                numerical_owner_AST_unchanged=True, model_calls=0, optimizer_updates=0)


def verify_same_owner_resource_tail(worker):
    import hashlib
    import inspect
    import json
    from pathlib import Path
    import psutil
    directory = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dtpo-exponential-credit-variance-20261017-v6/critic-rpc-host-cache-release-20261017-v2')
    facts = json.loads((directory / 'prepared.json').read_bytes())
    critic, = [item for item in worker.worker_dict.values() if getattr(item, 'critic', None) is not None]
    raw = inspect.unwrap(critic.update_critic.__func__)
    return dict(pid=psutil.Process().pid, birth=psutil.Process().create_time(),
                owner=str(Path(raw.__code__.co_filename)),
                owner_SHA256=hashlib.sha256(Path(facts['owner']).read_bytes()).hexdigest(),
                live_code_SHA256=hashlib.sha256(raw.__code__.co_code).hexdigest(),
                critic_scheduler_epoch=critic.critic_lr_scheduler.last_epoch,
                critic_optimizer_state_entries=len(critic.critic_optimizer.state))


ray.init(address=host + ':' + port, log_to_driver=False)
try:
    actors = [a for a in ray.util.list_named_actors(all_namespaces=True)
              if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(actors) == 2
    handles = [ray.get_actor(a['name'], namespace=a['namespace']) for a in actors]
    applied = ray.get([a.execute_with_func_generator.remote(func=install_same_owner_resource_tail) for a in handles])
    assert len({r['patched_code_SHA256'] for r in applied}) == 1
    assert owner.resolve().is_relative_to(ROOT / 'candidates/textcraft-PPO-host-memory-recovery-20261017-v3')
    pending = owner.with_name('fsdp_workers.py.critic-cache-v2.tmp')
    pending.write_bytes((PATCH / 'patched-fsdp_workers.py').read_bytes())
    os.replace(pending, owner)
    verified = ray.get([a.execute_with_func_generator.remote(func=verify_same_owner_resource_tail) for a in handles])
    assert all(r['owner_SHA256'] == prepared['patched_SHA256'] for r in verified)
    assert all(r['live_code_SHA256'] == applied[0]['patched_code_SHA256'] for r in verified)
    record = dict(status='completed_runtime_owner_binding_verified', collected=time.time(),
                  task='TextCraft', method='ppo', resource_version='critic-rpc-host-cache-release-20261017-v2',
                  upstream_commit=UPSTREAM, local_patch_commit=commit, original_source_receipt=job['source_receipt'],
                  deployment_source=str(Path(__file__)), deployment_source_SHA256=digest(Path(__file__).read_bytes()),
                  original_owner_SHA256=EXPECTED, source_bindings={str(owner): prepared['patched_SHA256']},
                  actual_driver=dict(pid=driver.pid, birth=driver.create_time()), applied=applied, verified=verified,
                  model_calls=0, optimizer_updates=0, training_method_changes=0,
                  automatic_new_critic_return_hook_executed=False,
                  scope='Actual owner function patched after original offload; first future native critic completion must verify execution')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    for name in ('formal-training.json', 'active-training.json', 'active-source.json',
                 'receipts/appworld-formal-training-20261017-v1/formal-training.json'):
        path = ROOT / name
        data = json.loads(path.read_bytes())
        for row in data['jobs']:
            if row.get('task') == 'TextCraft' and row.get('method') == 'ppo':
                assert (row['pid'], row['birth']) == (driver.pid, driver.create_time())
                row.setdefault('completed_runtime_overrides', []).append(dict(
                    status=record['status'], receipt=str(receipt), source_bindings=record['source_bindings'],
                    local_patch_commit=commit, resource_version=record['resource_version']))
        path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(status=record['status'], receipt=str(receipt), source_bindings=record['source_bindings'],
                          verified=verified, automatic_hook_executed=False)))
finally:
    ray.shutdown()
