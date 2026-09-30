"""Freeze the tested request transport over the current AppWorld entry.

No model, job, active manifest, or checkpoint is changed. All numerical owner
files and task parameters are inherited from the currently running release.
"""
from pathlib import Path
import hashlib
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, REPO, SCP, SSH

revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
fix_revision = subprocess.check_output(['git','rev-parse','64e6377'],cwd=REPO,text=True).strip()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='AppWorld')
source=read(Path(job['output'])/'source.json')
prior_path=Path(source['prepared_receipt']);prior=read(prior_path)
receipt=root/'receipts/owner-b8-dispatch-20260930/appworld-request-dispatch'
tested_path=receipt/'verified-candidate.json';tested=read(tested_path)
assert tested['test_returncode']==0
assert job['pid']==tested['active_driver'] and job['entry']==tested['active_entry']
assert sha(Path(tested['candidate_path']))==tested['candidate_sha256']
assert sha(receipt/'after-import-bound.xml')==tested['tests']['after_import_bound']
old=Path(job['entry']);entry=root/'candidates/appworld-request-dispatch-20261001/entry'
assert not entry.parent.exists(), 'Keep prepared versions immutable'
for name,expected in source['entry_sha256'].items():assert sha(old/name)==expected,name
for name,expected in source['verl_sha256'].items():assert sha(Path(job['verl_root'])/name)==expected,name
for name,expected in prior['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/name)==expected,name
shutil.copytree(old,entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(tested['candidate_path'],entry/'loop_owner_rollout.py')
actual={name:sha(entry/name) for name in source['entry_sha256']}
assert [n for n in actual if actual[n]!=source['entry_sha256'][n]]==['loop_owner_rollout.py']
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           VERL_ROOT=job['verl_root'],DT_ENTRY_ROOT=str(entry),LOOP_ROOT=prior['loop_root'])
env['PYTHONPATH']=':'.join([str(entry),job['verl_root'],prior['loop_root'],env['PYTHONPATH']])
nodes=['test_appworld_formal_workload_and_native_validator',
       'test_appworld_iteration_carrier_accepts_original_trainer_pop',
       'test_appworld_resume_forwards_only_original_checkpoint_options']
subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
    *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
    '--junitxml='+str(receipt/'cpu-tests.xml')],env=env,cwd=receipt,check=True)
record=dict(prior,prepared_unix=time.time(),status='prepared_not_launched',
    entry=str(entry),verl_root=job['verl_root'],dt_root=job['dt_root'],
    prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    future_checkpoint_root=job['checkpoints'],
    prior_prepared_receipt=str(prior_path),prior_prepared_receipt_sha256=sha(prior_path),
    preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
    request_dispatch_commit=@FIX_REVISION@,
    request_dispatch_receipt=str(tested_path),request_dispatch_receipt_sha256=sha(tested_path),
    scope='Only tested request transport changed over current frozen entry; native checkpoint restore and all model/task/training parameters preserved.',
    entry_sha256=actual,owner_head_sha256=source['owner_head_sha256'])
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','future_checkpoint_root']},indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY)
   .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha))
   .replace('@FIX_REVISION@',repr(fix_revision)))
local=AUDIT/'appworld-request-dispatch-20261001'
for name in ['prepared.json','cpu-tests.xml']:
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/appworld-request-dispatch/{name}',str(local/name)],check=True)
