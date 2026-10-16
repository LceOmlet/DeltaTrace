"""Pass current formal identity to the existing read-only VERL metadata RPC."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'
method = sys.argv[1]
assert method in ('dt', 'ppo')
phase = 'after-first-DT-v1' if '--after-DT' in sys.argv else 'formal-v1'
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
job, = [j for j in manifest['jobs'] if j.get('task') == 'AppWorld' and j.get('method') == method]
source = json.loads(Path(job['source_receipt']).read_bytes())
if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    command = [env['VENV_PYTHON'], '-X', 'utf8', __file__, method, '--client']
    if '--after-DT' in sys.argv:
        command.append('--after-DT')
    raise SystemExit(subprocess.call(command, env=env))

import psutil
import ray

# Reuse the owner metadata function, including the existing critic extension.
# Neither the original function body nor any training method is reimplemented.
wrapper = ROOT / 'receipts/appworld-native-PPO-32k-speed-20261016-v3/source/inspect_worker_imports.py'
wrapper_tree = ast.parse(wrapper.read_text())
assert isinstance(wrapper_tree.body[-1], ast.Expr)
assert isinstance(wrapper_tree.body[-1].value, ast.Call)
assert isinstance(wrapper_tree.body[-1].value.func, ast.Name)
assert wrapper_tree.body[-1].value.func.id == 'exec'
namespace = {}
exec(compile(ast.Module(body=wrapper_tree.body[:-1], type_ignores=[]), str(wrapper), 'exec'), namespace)
function, = [node for node in ast.parse(namespace['code']).body
             if isinstance(node, ast.FunctionDef) and node.name == 'inspect_worker']
BINDINGS = source['source_bindings']
exec(compile(ast.Module(body=[function], type_ignores=[]), str(wrapper), 'exec'), globals())

driver = psutil.Process(job['pid'])
assert driver.create_time() == job['birth']
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
request = AUDIT / ('worker-imports-request-' + method + '-' + phase + '.json')
request.write_text(json.dumps(dict(pid=os.getpid(), birth=psutil.Process().create_time(),
                                  job_pid=job['pid'], job_birth=job['birth'],
                                  status='read_only_rpc_pending', collected=time.time()), indent=2) + '\n')
ray.init(address=host + ':' + port, log_to_driver=False)
try:
    named = [x for x in ray.util.list_named_actors(all_namespaces=True)
             if 'WorkerDict_' in x['name'] and 'register_center' not in x['name']]
    assert len(named) == 2
    results = ray.get([ray.get_actor(x['name'], namespace=x['namespace']).execute_with_func_generator.remote(func=inspect_worker)
                       for x in named])
    assert all(m.get('matches_prepared_source') is not False for r in results for m in r['modules'].values())
    receipt = Path(job['output']).parent / 'runtime-overrides' / ('actual-worker-imports-' + phase + '.json')
    receipt.parent.mkdir(exist_ok=True)
    assert not receipt.exists(), 'Retain the previous actual import observation'
    record = dict(status='completed_read_only_worker_binding_check', task='AppWorld', method=method,
                  version=job['version'], local_patch_commit=source['local_patch_commit'],
                  collected_unix=time.time(), job=dict(pid=job['pid'], birth=job['birth']),
                  metadata_owner_wrapper=str(wrapper),
                  metadata_owner_wrapper_SHA256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                  composition_script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results=results, model_calls=0, DT_calls=0, optimizer_updates=0,
                  scope='Unchanged original metadata RPC; absent lazy modules remain unverified')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    request.write_text(json.dumps(dict(status=record['status'], receipt=str(receipt),
                                      completed=time.time()), indent=2) + '\n')
    print(json.dumps(dict(receipt=str(receipt), results=results)))
finally:
    ray.shutdown()
