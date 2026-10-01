"""Freeze SQL dispatch/scope fixes on its actual release; no job is changed."""
import hashlib
from pathlib import Path
import subprocess
from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

receipt=ROOT+'/receipts/owner-b8-dispatch-20260930/sql-rollout-scope'
names=['launch_sql_native.py','launch_textcraft_native.py','test_owner_entry_launch.py',
       'dt_training_batch.py','test_distributed_credit.py']
remote(f'mkdir -p {receipt}\n')
subprocess.run(SCP+[str(REPO/'experiments/rl'/n) for n in names]+[f'{SSH[-1]}:{receipt}/'],check=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,xml.etree.ElementTree as ET
root=Path('@ROOT@');receipt=Path('@RECEIPT@')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='SkyRL-SQL')
source=read(Path(job['output'])/'source.json');launch=read(Path(job['output'])/'launch.json')
old_entry=Path(job['entry']);old_owner=Path(job['verl_root'])
for n,h in source['entry_sha256'].items():assert sha(old_entry/n)==h,n
for n,h in source['verl_sha256'].items():assert sha(old_owner/n)==h,n
proof=root/'receipts/owner-b8-dispatch-20260930/rollout-scope'
prepared=read(proof/'prepared.json');identity=read(proof/'source-version.json')
replay=read(proof/'replay-complete.json')
assert all(r['status']=='passed_original_vllm_hybrid_comparison' and r['restored'] for r in replay['workers'])
assert sha(proof/'replay-complete.json')==identity['replay_sha256']
base=root/'candidates/sql-rollout-scope-20261001'
entry=base/'entry';owner=base/'verl'
assert not (receipt/'prepared.json').exists(), 'Preparation already completed; inspect its receipt'
fresh=not base.exists()
if fresh:
 shutil.copytree(old_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copytree(old_owner,owner,ignore=shutil.ignore_patterns('__pycache__','.git'))
 for n in @NAMES@:shutil.copy2(receipt/n,entry/n)
else:
 # The first CPU run imported the transport staging directory as cwd.
 # Recheck these frozen bytes from their real entry; do not replace them.
 for n in @NAMES@:assert sha(entry/n)==sha(receipt/n),n
for n,h in identity['files'].items():
 assert sha(proof/n)==h,n
 if fresh:shutil.copy2(proof/n,entry/n)
 else:assert sha(entry/n)==h,n
changed_owner={}
for relative,name in [('verl/workers/fsdp_workers.py','fsdp_workers.py'),
 ('agent_system/multi_turn_rollout/rollout_loop.py','rollout_loop.py')]:
 expected=next(h for p,h in prepared['original_sources'].items() if p.endswith('/'+relative))
 assert sha(old_owner/relative)==expected,relative
 assert sha(proof/name)==prepared['candidate_sources'][str(proof/name)]
 if fresh:shutil.copy2(proof/name,owner/relative)
 else:assert sha(owner/relative)==sha(proof/name),relative
 changed_owner[relative]=sha(owner/relative)
lock=read(entry/'verified_runtime.json')
for n,h in lock['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/n)==h,n
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),SCOPE_OWNER_ROOT=str(old_owner))
env['PYTHONPATH']=':'.join([str(entry),str(owner),env['PYTHONPATH']])
nodes=['test_two_rank_config_satisfies_native_validator[SkyRL-SQL-dt-formal]',
 'test_native_resume_keeps_original_workload']
xml=receipt/'cpu-tests.xml'
tests_passed=False
if xml.is_file():
 suite=ET.parse(xml).getroot().find('testsuite')
 tests_passed=(suite.get('tests')=='16' and all(suite.get(k)=='0' for k in ['errors','failures','skipped']))
 if not tests_passed:
  assert not (receipt/'cpu-tests-initial-import-path.xml').exists()
  shutil.copy2(xml,receipt/'cpu-tests-initial-import-path.xml')
if not tests_passed:
 subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
  *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
  str(entry/'test_distributed_credit.py'),str(entry/'test_owner_rollout_scope.py'),
  '--junitxml='+str(xml)],env=env,cwd=entry,check=True)
# Resolve the frozen launch with the same arguments; only its file location
# may differ. This is configuration construction, not model/environment setup.
code="""import json,sys
from pathlib import Path
from types import SimpleNamespace
from launch_sql_native import command
old=json.loads(Path(sys.argv[1]).read_text())
opts=old['options']
_,current=command(SimpleNamespace(method='dt',phase='formal',
 data=str(Path(opts['data.train_files']).parent),output=str(Path(sys.argv[1]).parent)))
print(json.dumps(current))
"""
options=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',code,str(Path(job['output'])/'launch.json')],env=env,cwd=entry,text=True))
changes={k:[launch['options'].get(k),options.get(k)] for k in launch['options'].keys()|options.keys()
 if launch['options'].get(k)!=options.get(k)}
assert set(changes)=={'data.custom_cls.path','+data.sql_chat_template'},changes
for before,after in changes.values():assert sha(Path(before))==sha(Path(after)),(before,after)
record=dict(prepared_unix=time.time(),status='prepared_not_launched',
 entry=str(entry),verl_root=str(owner),dt_root=job['dt_root'],prior_driver_pid=job['pid'],
 prior_entry=job['entry'],prior_verl_root=job['verl_root'],future_checkpoint_root=job['checkpoints'],
 prior_source_receipt=str(Path(job['output'])/'source.json'),prior_source_sha256=sha(Path(job['output'])/'source.json'),
 preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
 rollout_scope_commit=identity['code_commit'],rollout_scope_comparison=str(proof/'replay-complete.json'),
 rollout_scope_comparison_sha256=sha(proof/'replay-complete.json'),
 dt_dispatch_commit='4c0cbdd',configuration_changes=changes,
 cpu_tests=dict(path=str(xml),sha256=sha(xml),passed=16,
  unchanged_candidate_receipt_reused=tests_passed),
 entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
 owner_sha256={n:sha(owner/n) for n in source['verl_sha256']},changed_owner_sha256=changed_owner,
 dt_source_sha256=lock['dt_source_sha256'],
 scope='Original completed-checkpoint resume options; tested owner scope and official DT partitioner. No live job, sampler, PPO setting, model, optimizer or checkpoint changed.')
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','configuration_changes']},indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@RECEIPT@',receipt)
 .replace('@NAMES@',repr(names)).replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha)))
out=AUDIT/'sql-rollout-scope-20261001';out.mkdir(exist_ok=True)
for name in ['prepared.json','cpu-tests.xml']:
 subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}',str(out/name)],check=True)
