"""Read the recovered PPO through the unchanged native actor/critic metadata RPC."""
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
CANONICAL = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json'
job = json.loads(CANONICAL.read_bytes())
assert job['execution_version'] == 'textcraft-PPO-global-OOM-recovery-20261017-v1'
prepared = json.loads(Path(job['resume_launch_receipt']).read_bytes())
if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(prepared['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],'-X','utf8',__file__,'--client'],env=env))

import psutil
import ray

driver = psutil.Process(job['pid'])
assert driver.create_time() == job['birth']
children = driver.children(recursive=True)
workers = [p for p in children if p.name().startswith('ray::WorkerDict')]
if len(workers) != 2 or any('actor_rollout_generate_sequences' not in p.name() for p in workers):
    print(json.dumps(dict(status='pending_original_native_resume_initialization',pid=job['pid'],birth=job['birth'],
                          phases=[p.name() for p in workers],model_calls=0,optimizer_updates=0)))
    raise SystemExit(0)
owner = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/verify_native_PPO_bindings_v8.py'
function, = [n for n in ast.parse(owner.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == 'read']
exec(compile(ast.Module(body=[function],type_ignores=[]),str(owner),'exec'),globals())
gcs = next(p for p in children if p.name() == 'gcs_server')
port = next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host = next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
ray.init(address=host+':'+port,log_to_driver=False)
try:
    named = [a for a in ray.util.list_named_actors(all_namespaces=True)
             if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(named) == 2
    results = ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=read)
                       for a in named])
finally:
    ray.shutdown()
expected = json.loads((AUDIT/'TextCraft-PPO-native-OOM-recovery-input.json').read_bytes())
saved = {(r['role'],r['rank']):r for r in expected['saved_role_states']}
comparisons = []
for worker in results:
    assert worker['core']['SHA256'] == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    for role in worker['roles']:
        kind = 'actor' if 'actor' in role else 'critic'
        original = saved[(kind,role['rank'])]
        comparisons.append(dict(role=kind,rank=role['rank'],
            scheduler_matches=role['scheduler_last_epoch'] == original['scheduler'],
            optimizer_steps_match=role['optimizer_state_steps'] == original['optimizer_steps'],
            optimizer_entries_match=role['optimizer_state_entries'] == original['optimizer_entries'],
            rollout_RNG_owner_matches=kind!='actor' or (role['rollout_rng_restored_variable_present']
                                                       and role['rollout_rng_owner_is_native_manager'])))
assert len(comparisons) == 4
matches = all(all(v for k,v in row.items() if k not in ('role','rank')) for row in comparisons)
receipt = Path(job['run'])/'runtime-overrides/native-PPO-restore-global-OOM-20261017-v1.json'
assert not receipt.exists()
record = dict(status='verified_original_native_PPO_checkpoint_restore' if matches else 'native_state_readback_did_not_match_initial_checkpoint_window',
              collected=time.time(),job=dict(pid=job['pid'],birth=job['birth'],execution_version=job['execution_version']),
              checkpoint=expected['checkpoint'],completed_iteration=expected['completed_iteration'],
              data_SHA256=expected['data_SHA256'],workers=results,comparisons=comparisons,
              reader_path=str(owner),reader_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(),
              model_calls=0,optimizer_updates=0,checkpoint_create_calls=0,
              scope='Original native load restores all checkpoint components; metadata read compares saved actor/critic scheduler and optimizer state and native rollout RNG owner. Post-sampling RNG equality or future trajectory equality not claimed.')
receipt.write_text(json.dumps(record,indent=2)+'\n')
if matches:
    job.update(status=record['status'],actual_native_restore_receipt=str(receipt),
               actual_native_binding_receipt=str(receipt),actual_native_restore_observed_unix=record['collected'])
    CANONICAL.write_text(json.dumps(job,indent=2)+'\n')
    for name in ('formal-training.json','active-training.json'):
        path = ROOT/name
        manifest = json.loads(path.read_bytes())
        row, = [j for j in manifest['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='ppo']
        assert (row['pid'],row['birth']) == (job['pid'],job['birth'])
        row.update(status=record['status'],actual_native_restore_receipt=str(receipt),actual_native_binding_receipt=str(receipt),
                   ray_session=str(Path(next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--log_dir='))).parent))
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    path=ROOT/'active-source.json'
    manifest=json.loads(path.read_bytes())
    manifest['TextCraft_PPO_global_OOM_recovery_20261017_v1'].update(status=record['status'],actual_native_restore_receipt=str(receipt))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],receipt=str(receipt),comparisons=comparisons)))
