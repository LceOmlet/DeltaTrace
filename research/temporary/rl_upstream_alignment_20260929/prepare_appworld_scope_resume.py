"""Freeze the tested native rollout scope over the actual AppWorld release.

No active process, manifest, checkpoint or task parameter is changed. The
original submission helper will pass a completed checkpoint to VERL itself.
"""
import hashlib
from pathlib import Path
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, REPO, SCP, SSH

revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='AppWorld')
source=read(Path(job['output'])/'source.json')
prior_path=Path(source['prepared_receipt']);prior=read(prior_path)
tested=root/'receipts/owner-b8-dispatch-20260930/rollout-scope'
proof=read(tested/'prepared.json');identity=read(tested/'source-version.json')
replay=read(tested/'replay-complete.json')
assert proof['cpu_test_returncode']==0
assert all(w['status']=='passed_original_vllm_hybrid_comparison' and w['restored'] for w in replay['workers'])
assert sha(tested/'prepared.json')==identity['prepared_sha256']
assert sha(tested/'replay-complete.json')==identity['replay_sha256']
old_entry=Path(job['entry']);old_owner=Path(job['verl_root'])
for name,expected in source['entry_sha256'].items():assert sha(old_entry/name)==expected,name
for name,expected in source['verl_sha256'].items():assert sha(old_owner/name)==expected,name
for name,expected in prior['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/name)==expected,name
names={'verl/workers/fsdp_workers.py':'fsdp_workers.py',
       'agent_system/multi_turn_rollout/rollout_loop.py':'rollout_loop.py'}
for relative,name in names.items():
    old_expected=next(v for k,v in proof['original_sources'].items() if k.endswith('/'+relative))
    assert sha(old_owner/relative)==old_expected,relative
    assert sha(tested/name)==proof['candidate_sources'][str(tested/name)],name
for name,expected in identity['files'].items():assert sha(tested/name)==expected,name
base=root/'candidates/appworld-rollout-scope-20261001'
receipt=root/'receipts/owner-b8-dispatch-20260930/appworld-rollout-scope'
assert not base.exists() and not receipt.exists(), 'Inspect the existing frozen preparation'
receipt.mkdir(parents=True)
entry=base/'entry';owner=base/'verl'
shutil.copytree(old_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(old_owner,owner,ignore=shutil.ignore_patterns('__pycache__','.git'))
for relative,name in names.items():shutil.copy2(tested/name,owner/relative)
for name in identity['files']:shutil.copy2(tested/name,entry/name)
actual={p.name:sha(p) for p in entry.glob('*.py')}
assert all(actual[name]==expected for name,expected in source['entry_sha256'].items())
changed=[name for name,expected in source['verl_sha256'].items() if sha(owner/name)!=expected]
assert set(changed)==set(names),changed
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),LOOP_ROOT=prior['loop_root'],SCOPE_OWNER_ROOT=str(old_owner))
env['PYTHONPATH']=':'.join([str(entry),str(owner),prior['loop_root'],env['PYTHONPATH']])
nodes=['test_appworld_formal_workload_and_native_validator',
       'test_appworld_iteration_carrier_accepts_original_trainer_pop',
       'test_appworld_resume_forwards_only_original_checkpoint_options']
subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
    *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
    str(entry/'test_owner_rollout_scope.py'),
    '--junitxml='+str(receipt/'cpu-tests.xml')],env=env,cwd=receipt,check=True)
verified_owner={name:sha(owner/name) for name in set(prior['owner_head_sha256'])|set(names)}
record=dict(prior,prepared_unix=time.time(),status='prepared_not_launched',
    entry=str(entry),verl_root=str(owner),dt_root=job['dt_root'],
    prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    future_checkpoint_root=job['checkpoints'],
    prior_prepared_receipt=str(prior_path),prior_prepared_receipt_sha256=sha(prior_path),
    preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
    rollout_scope_commit=identity['code_commit'],rollout_scope_comparison=str(tested/'replay-complete.json'),
    rollout_scope_comparison_sha256=sha(tested/'replay-complete.json'),
    scope='Original whole-collection sharding-manager scope only; all prior task, dispatch, actor/head and numerical repairs preserved. No job launched.',
    entry_sha256=actual,owner_head_sha256=verified_owner)
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','future_checkpoint_root']},indent=2))
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
   .replace('@REVISION@', repr(revision)).replace('@SCRIPT_SHA@', repr(script_sha)))
local=AUDIT/'appworld-rollout-scope-20261001';local.mkdir(exist_ok=True)
for name in ['prepared.json','cpu-tests.xml']:
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/{name}',str(local/name)],check=True)
