set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,psutil,shutil,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-efficiency-20261007/fresh-official-padding-v1');candidate=root/'candidates/appworld-fresh-official-padding-20261007-v1'
assert not candidate.exists(), 'Never overwrite prepared sources'
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
source=Path(job['source_receipt']);s=json.loads(source.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for base,hashes in [(Path(job['entry']),s['entry_sha256']),(Path(job['verl_root']),s['owner_head_sha256'])]:
 for name,digest in hashes.items():assert sha(base/name)==digest,(base,name)
shutil.copytree(job['entry'],candidate/'entry',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copytree(job['verl_root'],candidate/'verl',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
sys.path.insert(0,str(out));sys.path.insert(1,str(candidate/'entry'))
from patch_actor_shared_right_padding import patch
actor=candidate/'verl/verl/workers/actor/dp_actor.py';before=actor.read_bytes();after=patch(before.decode()).encode()
assert hashlib.sha256(before).hexdigest()=='1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd'
assert hashlib.sha256(after).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
actor.write_bytes(after)
launcher=candidate/'entry/launch_appworld_native.py';old=launcher.read_bytes();needle=b"'trainer.resume_mode': 'auto'"
assert old.count(needle)==1;launcher.write_bytes(old.replace(needle,b"'trainer.resume_mode': 'disable'"))
trainer=candidate/'verl/verl/trainer/ppo/ray_trainer.py'
assert sha(trainer)=='d35ddd26b7497b153cec22f22e92eb1a08ef70385428dcbdcc37e879d9f16a4f'
assert 'verl_F.masked_whiten(data.batch["dt_token_advantages"], response_mask)' in trainer.read_text()
env=dict(os.environ,VERL_ROOT=str(candidate/'verl'),DT_ENTRY_ROOT=str(candidate/'entry'),LOOP_ROOT=s['loop_root'],
 APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'),
 PYTHONPATH=':'.join([str(out),str(candidate/'entry'),str(candidate/'verl'),s['pythonpath']]))
with (out/'cpu-validator.log').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(out/'test_owner_entry_launch.py')+'::test_appworld_formal_workload_and_native_validator',
  '--junitxml='+str(out/'cpu-validator.xml')],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
entry_hashes={str(p.relative_to(candidate/'entry')):sha(p) for p in (candidate/'entry').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
owner_hashes={str(p.relative_to(candidate/'verl')):sha(p) for p in (candidate/'verl').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
proof=root/'receipts/owner-b8-dispatch-20260930/actor-shared-right-padding-20261004-v4/result.json';proof_data=json.loads(proof.read_text())
assert proof_data['phase']=='complete' and proof_data['padding_assertion_passed']
assert all(x['actor_sha256']==sha(actor) for x in proof_data['selected_owners'])
record=dict(status='prepared_only_not_deployed',observed_unix=time.time(),base_source=str(source),base_source_sha256=sha(source),
 candidate=str(candidate),entry=str(candidate/'entry'),verl_root=str(candidate/'verl'),dt_root=job['dt_root'],loop_root=s['loop_root'],
 entry_sha256=entry_hashes,owner_sha256=owner_hashes,changed_entry=['launch_appworld_native.py'],changed_owner=['verl/workers/actor/dp_actor.py'],
 official_padding_receipt=str(proof),official_padding_sha256=sha(proof),actor_sha256=sha(actor),trainer_sha256=sha(trainer),
 cpu_validator_returncode=result.returncode,checkpoint_resume_mode='disable',checkpoint_restore_requested=False,
 unchanged='DT d/Q/V/A, raw values, approved full-collected-batch whitening, PPO loss/defaults, official task budget, LoRA8/16, per-card B4, 32k',
 prepared_files={str(p):sha(p) for p in out.glob('*.py')})
(out/'prepared.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['entry_sha256','owner_sha256','prepared_files']}),flush=True)
if result.returncode:print((out/'cpu-validator.log').read_text()[-3000:])
result.check_returncode()
PY
