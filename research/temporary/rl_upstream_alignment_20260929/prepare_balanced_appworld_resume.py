"""Freeze and CPU-check a resume candidate; never stop or launch a trainer."""
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, REPO, AUDIT, SCP, SSH

receipt=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-balanced-resume'
remote(f'mkdir -p {receipt}\n')
names=('owner_runtime_options.py','launch_appworld_native.py','dt_training_batch.py',
       'patch_actor_fused_head.py','patch_actor_entropy_dispatch.py','test_owner_entry_launch.py',
       'test_distributed_credit.py')
subprocess.run(SCP+[str(REPO/'experiments/rl'/name) for name in names]+
               [f'{SSH[-1]}:{receipt}/'],check=True)
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path('@ROOT@');receipt=Path('@RECEIPT@')
active=json.loads((root/'active-training.json').read_text())
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
base=root/'candidates/appworld-balanced-resume-20261001'
entry=base/'entry';verl=base/'verl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not base.exists():
    shutil.copytree(old['entry'],entry,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(old['verl_root'],verl,ignore=shutil.ignore_patterns('__pycache__','.git'))
    for name in @NAMES@:shutil.copy2(receipt/name,entry/name)
    for name in ('patch_actor_fused_head.py','patch_actor_entropy_dispatch.py'):
        subprocess.run([os.environ['VENV_PYTHON'],str(entry/name),str(verl)],check=True)
else:
    # Reuse this exact unlaunched candidate after the test selector repair;
    # never replace already frozen bytes with newer workspace contents.
    assert all(j['entry']!=str(entry) for j in active['jobs'])
    for name in @NAMES@:assert sha(entry/name)==sha(receipt/name),name
print('Frozen candidate bytes ready; validating existing owner head hashes.',flush=True)
tested=root/'candidates/official-verl-20bd331-fused-head-b4-20260930'
head_files=('verl/utils/experimental/torch_functional.py','verl/models/transformers/monkey_patch.py',
            'verl/models/transformers/qwen3_vl.py','verl/workers/actor/dp_actor.py')
for name in head_files:assert sha(verl/name)==sha(tested/name),name
lock=json.loads((entry/'verified_runtime.json').read_text())
for name,value in lock['dt_source_sha256'].items():
    assert sha(Path(os.environ['DT_ROOT'])/name)==value,name
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           VERL_ROOT=str(verl),DT_ENTRY_ROOT=str(entry),LOOP_ROOT=old['loop_root'])
env['PYTHONPATH']=':'.join([str(entry),str(verl),old['loop_root'],env['PYTHONPATH']])
prior=receipt/'launch-cpu-tests.xml'
if prior.exists() and not (receipt/'launch-cpu-tests-unscoped.xml').exists():
    shutil.copy2(prior,receipt/'launch-cpu-tests-unscoped.xml')
# -k also matches parent directory names, so select the three exact node IDs.
nodes=['test_appworld_formal_workload_and_native_validator',
       'test_appworld_iteration_carrier_accepts_original_trainer_pop',
       'test_appworld_resume_forwards_only_original_checkpoint_options']
print('Running exact AppWorld config/carrier/resume tests on CPU.',flush=True)
result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',
    *[str(entry/'test_owner_entry_launch.py')+'::'+name for name in nodes],
    '--junitxml='+str(receipt/'launch-cpu-tests.xml')],env=env,cwd=str(receipt))
result.check_returncode()
result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(entry/'test_distributed_credit.py'),
    '--junitxml='+str(receipt/'dispatch-cpu-tests.xml')],env=env,cwd=str(receipt))
result.check_returncode()
record=dict(prepared_unix=time.time(),status='prepared_not_launched',
    scope='No current process, active manifest, frozen source, model, optimizer or checkpoint changed.',
    prior_driver_pid=old['pid'],prior_entry=old['entry'],prior_verl_root=old['verl_root'],
    entry=str(entry),verl_root=str(verl),loop_root=old['loop_root'],dt_root=old['dt_root'],
    future_checkpoint_root=old['checkpoints'],
    resume_behavior='Original VERL resume_path restores model/optimizer/global step/dataloader; select only a completed original marker at launch, not now.',
    entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
    owner_head_sha256={name:sha(verl/name) for name in head_files},
    dt_source_sha256=lock['dt_source_sha256'])
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','future_checkpoint_root']},indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@RECEIPT@',receipt).replace('@NAMES@',repr(names)))
local=AUDIT/'appworld-balanced-resume-20261001';local.mkdir(exist_ok=True)
for name in ('prepared.json','launch-cpu-tests.xml','dispatch-cpu-tests.xml'):
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}',str(local/name)],check=True)
