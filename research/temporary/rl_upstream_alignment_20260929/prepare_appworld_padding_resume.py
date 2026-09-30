"""Compose two frozen, verified candidates for the original checkpoint loader.

No live files, processes, manifests, models, or task parameters are changed.
"""
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, SCP, SSH

receipt=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-balanced-padding-resume'
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,time
root=Path('@ROOT@');receipt=Path('@RECEIPT@')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
previous_path=root/'receipts/owner-b8-dispatch-20260930/appworld-balanced-resume/prepared.json'
previous=read(previous_path)
padding_receipt=root/'receipts/owner-b8-dispatch-20260930/actor-response-padding'
padding=read(padding_receipt/'candidate.json')
owner_test=read(padding_receipt/'real-model/owner-padding-assertion.json')
assert owner_test['passed'] and padding['test_returncode']==0
base=root/'candidates/appworld-balanced-padding-resume-20261001'
assert not base.exists() and not receipt.exists(), 'Never overwrite a frozen attempt'
prior_entry=Path(previous['entry']);prior_verl=Path(previous['verl_root'])
for name,digest in previous['entry_sha256'].items():assert sha(prior_entry/name)==digest,name
for name,digest in previous['owner_head_sha256'].items():assert sha(prior_verl/name)==digest,name
actor='verl/workers/actor/dp_actor.py';tested=Path(padding['candidate'])
assert sha(prior_verl/actor)==padding['before_sha256']
assert sha(tested/actor)==padding['after_sha256']
def methods(path):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor')
    return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
before,after=methods(prior_verl/actor),methods(tested/actor)
assert before.keys()==after.keys()
assert [n for n in before if before[n]!=after[n]]==['_forward_micro_batch']
receipt.mkdir(parents=True)
entry=base/'entry';verl=base/'verl'
shutil.copytree(prior_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(prior_verl,verl,ignore=shutil.ignore_patterns('__pycache__','.git'))
shutil.copy2(tested/actor,verl/actor)
for name,expected in padding['local_files_sha256'].items():
    assert sha(padding_receipt/name)==expected,name
    shutil.copy2(padding_receipt/name,entry/name)
for name,expected in previous['dt_source_sha256'].items():assert sha(Path(previous['dt_root'])/name)==expected,name
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           VERL_ROOT=str(verl),DT_ENTRY_ROOT=str(entry),LOOP_ROOT=previous['loop_root'])
env['PYTHONPATH']=':'.join([str(entry),str(verl),previous['loop_root'],env['PYTHONPATH']])
nodes=['test_appworld_formal_workload_and_native_validator',
       'test_appworld_iteration_carrier_accepts_original_trainer_pop',
       'test_appworld_resume_forwards_only_original_checkpoint_options']
subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',
    *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
    str(entry/'test_distributed_credit.py'),str(entry/'test_shared_padding.py'),
    '--junitxml='+str(receipt/'cpu-tests.xml')],env=env,cwd=receipt,check=True)
record=dict(previous,prepared_unix=time.time(),status='prepared_not_launched',
    entry=str(entry),verl_root=str(verl),
    prior_prepared_receipt=str(previous_path),prior_prepared_receipt_sha256=sha(previous_path),
    scope='Combines existing balanced dispatch and owner-tested padding seam; formal jobs and all task/LoRA/microbatch parameters unchanged.',
    entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
    owner_head_sha256={name:sha(verl/name) for name in previous['owner_head_sha256']},
    padding_comparison_receipt=str(padding_receipt/'real-model/owner-padding-assertion.json'),
    padding_comparison_receipt_sha256=sha(padding_receipt/'real-model/owner-padding-assertion.json'),
    padding_candidate_receipt=str(padding_receipt/'candidate.json'),
    padding_candidate_receipt_sha256=sha(padding_receipt/'candidate.json'))
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','future_checkpoint_root']},indent=2))
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@RECEIPT@',receipt))
local=AUDIT/'appworld-balanced-padding-resume-20261001';local.mkdir(exist_ok=True)
for name in ('prepared.json','cpu-tests.xml'):
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}',str(local/name)],check=True)
