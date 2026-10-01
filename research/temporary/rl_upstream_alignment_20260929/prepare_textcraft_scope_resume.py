"""Freeze TextCraft's effective B4 release plus already tested owner repairs.

Preparation only: the running job, sampler and checkpoints remain untouched.
The completed worker receipt owns effective settings, not the old launch file.
"""
import hashlib
from pathlib import Path
import subprocess
from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

receipt = ROOT+'/receipts/owner-b8-dispatch-20260930/textcraft-rollout-scope'
names = ['owner_runtime_options.py', 'launch_sql_native.py', 'launch_textcraft_native.py',
         'test_owner_entry_launch.py', 'dt_training_batch.py', 'test_distributed_credit.py']
remote(f'mkdir -p {receipt}\n')
subprocess.run(SCP+[str(REPO/'experiments/rl'/n) for n in names]+[f'{SSH[-1]}:{receipt}/'], check=True)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,psutil,shutil,subprocess,time,xml.etree.ElementTree as ET
root=Path('@ROOT@');receipt=Path('@RECEIPT@')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='TextCraft')
source_path=Path(job['output'])/'source.json';source=read(source_path)
launch_path=Path(job['output'])/'launch.json';launch=read(launch_path)
old_entry=Path(job['entry']);old_owner=Path(job['verl_root'])
for n,h in source['entry_sha256'].items():assert sha(old_entry/n)==h,n
for n,h in source['verl_sha256'].items():assert sha(old_owner/n)==h,n
overlay_path=root/'receipts/owner-b8-dispatch-20260930/live-TextCraft-complete.json'
overlay=read(overlay_path)
driver=psutil.Process(job['pid'])
assert abs(driver.create_time()-job['observed_process_created_unix'])<.02
workers={p.pid for p in driver.children(recursive=True) if 'WorkerDict' in p.name()}
assert len(overlay['workers'])==2 and {w['pid'] for w in overlay['workers']}==workers
for w in overlay['workers']:
 assert (w['actor_microbatch'],w['logprob_microbatch'],w['lora_rank'],w['lora_alpha'])==(4,4,8,16)
 for n,h in w['source_sha256'].items():assert sha(Path(w['source'])/n)==h,n
assert overlay['workers'][0]['source_sha256']==overlay['workers'][1]['source_sha256']
proof=root/'receipts/owner-b8-dispatch-20260930/rollout-scope'
scope=read(proof/'prepared.json');identity=read(proof/'source-version.json')
replay=read(proof/'replay-complete.json')
assert sha(proof/'replay-complete.json')==identity['replay_sha256']
assert all(w['status']=='passed_original_vllm_hybrid_comparison' and w['restored'] for w in replay['workers'])
padding_dir=root/'receipts/owner-b8-dispatch-20260930/actor-response-padding'
padding=read(padding_dir/'candidate.json')
padding_assertion=padding_dir/'real-model/owner-padding-assertion.json'
assert read(padding_assertion)['passed'] and padding['test_returncode']==0
actor='verl/workers/actor/dp_actor.py'
assert sha(old_owner/actor)==padding['before_sha256']
assert sha(Path(padding['candidate'])/actor)==padding['after_sha256']
base=root/'candidates/textcraft-rollout-scope-20261001'
assert not (receipt/'prepared.json').exists(), 'Inspect completed preparation'
fresh=not base.exists()
entry=base/'entry';owner=base/'verl'
if fresh:
 shutil.copytree(old_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copytree(old_owner,owner,ignore=shutil.ignore_patterns('__pycache__','.git'))
def freeze(src,dst):
 if fresh:shutil.copy2(src,dst)
 else:assert sha(src)==sha(dst),(src,dst)
for n in @NAMES@:freeze(receipt/n,entry/n)
for n,h in overlay['workers'][0]['source_sha256'].items():
 freeze(Path(overlay['workers'][0]['source'])/n,owner/n)
freeze(Path(padding['candidate'])/actor,owner/actor)
for n,h in padding['local_files_sha256'].items():
 assert sha(padding_dir/n)==h,n
 freeze(padding_dir/n,entry/n)
for n,h in identity['files'].items():
 assert sha(proof/n)==h,n
 freeze(proof/n,entry/n)
for relative,name in [('verl/workers/fsdp_workers.py','fsdp_workers.py'),
 ('agent_system/multi_turn_rollout/rollout_loop.py','rollout_loop.py')]:
 expected=next(h for p,h in scope['original_sources'].items() if p.endswith('/'+relative))
 assert sha(old_owner/relative)==expected,relative
 assert sha(proof/name)==scope['candidate_sources'][str(proof/name)]
 freeze(proof/name,owner/relative)
lock=read(entry/'verified_runtime.json')
for n,h in lock['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/n)==h,n
env=os.environ.copy()
agentgym=launch['options']['+env.textcraft']['owner_root']
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),SCOPE_OWNER_ROOT=str(old_owner),AGENTGYM_RL_ROOT=agentgym)
env['PYTHONPATH']=':'.join([str(entry),str(owner),env['PYTHONPATH']])
nodes=['test_textcraft_formal_owner_workload_and_native_validator','test_native_resume_keeps_original_workload[TextCraft]']
xml=receipt/'cpu-tests.xml'
if fresh:
 subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
  *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
  str(entry/'test_distributed_credit.py'),str(entry/'test_shared_padding.py'),str(entry/'test_owner_rollout_scope.py'),
  '--junitxml='+str(xml)],env=env,cwd=entry,check=True)
else:
 # Initial preparation passed all 49 tests, then incorrectly expected an
 # explicit false in the old launch where the default option was absent.
 # Only that receipt comparison changes; candidate bytes stay identical.
 suite=ET.parse(xml).getroot().find('testsuite')
 assert suite.get('tests')=='49' and all(suite.get(k)=='0' for k in ['errors','failures','skipped'])
code="""import json,sys
from pathlib import Path
from launch_textcraft_native import options_for
path=Path(sys.argv[1]);old=json.loads(path.read_text())
options,_=options_for(Path(old['options']['data.train_files']).parent,path.parent)
print(json.dumps(options))
"""
options=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',code,str(launch_path)],env=env,cwd=entry,text=True))
changes={k:[launch['options'].get(k),options.get(k)] for k in launch['options'].keys()|options.keys()
 if launch['options'].get(k)!=options.get(k)}
# These historical startup values were already superseded by the completed
# B4/head RPC. Every task, sample, reward, loss and budget option stays equal.
allowed={
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':[1,4],
 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':[1,4],
 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu':[1,4],
 'actor_rollout_ref.model.use_fused_kernels':[None,True],
 '+actor_rollout_ref.model.fused_kernel_options.impl_backend':[None,'torch']}
assert all(k in allowed and v==allowed[k] for k,v in changes.items()),changes
for k,v in allowed.items():assert options[k]==v[1],k
# Original LoRA reference uses actor compute_log_prob with adapters disabled;
# it does not load a second reference model or read the non-LoRA ref batch.
tree=ast.parse((owner/'verl/workers/fsdp_workers.py').read_text())
ref=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_ref_log_prob')
lora=next(n for n in ref.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self._is_lora')
assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='self.compute_log_prob' for n in ast.walk(lora))
suite=ET.parse(xml).getroot().find('testsuite')
record=dict(prepared_unix=time.time(),status='prepared_not_launched',entry=str(entry),verl_root=str(owner),
 dt_root=job['dt_root'],prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
 future_checkpoint_root=job['checkpoints'],prior_source_receipt=str(source_path),prior_source_sha256=sha(source_path),
 preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
 completed_overlay_receipt=str(overlay_path),completed_overlay_sha256=sha(overlay_path),
 rollout_scope_commit=identity['code_commit'],rollout_scope_comparison=str(proof/'replay-complete.json'),
 rollout_scope_comparison_sha256=sha(proof/'replay-complete.json'),dt_dispatch_commit='4c0cbdd',
 padding_comparison_receipt=str(padding_assertion),padding_comparison_receipt_sha256=sha(padding_assertion),
 configuration_changes=changes,configuration_change_scope='Only folds completed PID-bound B4/head overrides into startup; all task/workload options unchanged.',
 lora_reference_owner_path='compute_ref_log_prob -> _is_lora -> compute_log_prob; original method retained',
 cpu_tests=dict(path=str(xml),sha256=sha(xml),passed=int(suite.get('tests')),
  unchanged_candidate_receipt_reused=not fresh),
 entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
 owner_sha256={n:sha(owner/n) for n in source['verl_sha256']},
 changed_owner_sha256={n:sha(owner/n) for n,h in source['verl_sha256'].items() if sha(owner/n)!=h},
 dt_source_sha256=lock['dt_source_sha256'],
 scope='Original VERL resume; verified B4/head overlay, padding, DT partitioner and whole-rollout owner context. No live job changed.')
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','prior_driver_pid','configuration_changes','cpu_tests']},indent=2))
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY).replace('@RECEIPT@', receipt)
   .replace('@NAMES@', repr(names)).replace('@REVISION@', repr(revision)).replace('@SCRIPT_SHA@', repr(script_sha)))
out=AUDIT/'textcraft-rollout-scope-20261001';out.mkdir(exist_ok=True)
for name in ['prepared.json','cpu-tests.xml']:
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}',str(out/name)], check=True)
