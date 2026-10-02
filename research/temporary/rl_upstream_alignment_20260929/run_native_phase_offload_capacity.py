"""Reuse the verified B8 capacity fixture with VERL's phase-offload setting.

This is a single bounded diagnostic on idle GPUs 6/7, not a formal job or a
new numerical implementation. The original fixture, installed actor, loss,
optimizer, vLLM and kernels remain the owners. Only offload_policy is false;
param_offload/optimizer_offload and the fixed B4, rank8/alpha16 remain true.
"""
from pathlib import Path
import hashlib
import subprocess

from stage_environment_entry import AUDIT, ENTRY, ROOT, remote


FIXTURE = AUDIT / 'verify_owner_b8_capacity.py'
FIXTURE_SHA256 = 'dd7783122843033e98455e4178f2815bc65f6fbc8fa8757dcb65052dcfebbb60'


if __name__ == '__main__':
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('@ROOT@')
active=json.loads((root/'active-training.json').read_text())
actor_job=next(j for j in active['jobs'] if j['task']=='AppWorld')
entry_job=next(j for j in active['jobs'] if j['task']=='TextCraft')
for job in (actor_job,entry_job):
 assert psutil.Process(job['pid']).create_time()==job['observed_process_created_unix']
physical=subprocess.check_output(['mx-smi'],text=True)
assert not any(re.match(r'^\|\s*[67]\s+\d+\s+\S+',line) for line in physical.splitlines()), 'GPU6/7 have a current process'
out=root/'receipts/owner-b8-dispatch-20260930'/f'phase-offload-{int(time.time())}'
out.mkdir()
(out/'physical-before.txt').write_text(physical)
fixture=root/'receipts/owner-b8-dispatch-20260930/verify_owner_b8_capacity.py'
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='@FIXTURE_SHA@'
source=fixture.read_text()
edits=[
 ("OUT=ROOT/'receipts/owner-b8-dispatch-20260930'", "OUT=Path(os.environ['DT_CAPACITY_RECEIPT'])"),
 ('    import pyarrow.parquet as pq',
  '    cfg.actor_rollout_ref.actor.fsdp_config.offload_policy=False\n    import pyarrow.parquet as pq'),
 ('self.observations=dict(rank=self.rank,config=',
  'self.observations=dict(rank=self.rank,worker_pid=os.getpid(),phase_offload=self._is_offload_param,config='),
]
for old,new in edits:
 assert source.count(old)==1
 source=source.replace(old,new,1)
candidate=out/'native_b8_capacity.py'
compile(source,str(candidate),'exec')
candidate.write_text(source)
owner=Path(actor_job['verl_root']);entry=Path(entry_job['entry'])
pins={
 'verl/workers/actor/dp_actor.py':'1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd',
 'verl/utils/torch_functional.py':'079a6d20696a861340687d6e61fb4162cc1d436837ce747ac9ef318b738c1701',
 'verl/workers/fsdp_workers.py':'807e51856f9990d408fb2c33998fdcf9b0cbf3f230178677d6a5d2ecc0b7cc0d',
}
fingerprints={}
for name,expected in pins.items():
 p=owner/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==expected,(name,actual)
 fingerprints[str(p)]=actual
for name in ('verl/utils/fsdp_utils.py','verl/trainer/config/ppo_trainer.yaml'):
 p=owner/name;fingerprints[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('owner_runtime_options.py','launch_textcraft_native.py','owner_environment_configs.json','verified_runtime.json'):
 p=entry/name;fingerprints[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),CUDA_VISIBLE_DEVICES='6,7',
 DT_CAPACITY_RECEIPT=str(out),
 AGENTGYM_RL_ROOT=str(root/'runs/official-trajectory-20260930-v4/textcraft-owner'))
env['PYTHONPATH']=':'.join([str(entry),str(owner),env.get('PYTHONPATH','')])
argv=[env['VENV_PYTHON'],'-u',str(candidate)]
receipt=dict(scope='One original native B8x32768 capacity fixture with official per-phase offload, not a numerical tolerance or formal training claim',
 source_commit='@COMMIT@',driver_source_sha256='@SCRIPT_SHA@',original_fixture=str(fixture),
 original_fixture_sha256='@FIXTURE_SHA@',candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
 owner=str(owner),entry=str(entry),source_sha256=fingerprints,
 sole_configuration_change={'actor_rollout_ref.actor.fsdp_config.offload_policy':False},
 fixed={'lora_rank':8,'lora_alpha':16,'actor_microbatch_per_gpu':4,'global_batch':8,'total_context':32768},
 gpus=[6,7],formal_jobs_unchanged=[dict(task=j['task'],pid=j['pid'],created_unix=j['observed_process_created_unix']) for j in active['jobs']],argv=argv)
(out/'source.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (out/'capacity.log').open('wb') as f:
 p=subprocess.Popen(argv,env=env,cwd=str(out),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(receipt,pid=p.pid,started_unix=time.time(),created_unix=psutil.Process(p.pid).create_time())
(out/'job.json').write_text(json.dumps(job,indent=2)+'\n')
with (out/'resource-observer.log').open('wb') as f:
 observer=subprocess.Popen([env['VENV_PYTHON'],'@ENTRY@/observe_entry_resources.py','--job',str(out/'job.json'),
  '--output',str(out/'physical-resources.jsonl')],env=env,cwd=str(out),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(receipt=str(out),pid=p.pid,created_unix=job['created_unix'],observer=observer.pid,scope=receipt['scope'])))
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT)
           .replace('@FIXTURE_SHA@', FIXTURE_SHA256)
           .replace('@COMMIT@', commit).replace('@SCRIPT_SHA@', script_sha))
