"""Start the already CPU-checked actor comparison on idle GPUs 2/3.

User's cards 1/2 (physical GPUs 0/1) remain empty; TextCraft stays on 4/5.
"""
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, remote


if __name__ == '__main__':
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/actor-shared-right-padding-20261004-v4'
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import pathlib,json,hashlib,os,psutil,re,subprocess,time
root=pathlib.Path('@ROOT@');out=pathlib.Path('@OUT@')
assert not (out/'job.json').exists(), 'Reuse the recorded process; never duplicate a probe'
proof=json.loads((out/'prepared.json').read_bytes())
assert proof['test_returncode']==0
assert proof.get('factory_cpu_check'), 'Resolve the observed diagnostic factory error before using GPUs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
candidate=pathlib.Path(proof['candidate']);base=pathlib.Path(proof['base']);actor='verl/workers/actor/dp_actor.py'
assert sha(candidate/actor)==proof['after_actor_sha256'] and sha(base/actor)==proof['before_actor_sha256']
for path,digest in proof['files'].items():assert sha(pathlib.Path(path))==digest
physical=subprocess.check_output(['mx-smi'],text=True)
processes=physical.split('| Process:')[-1]
assert not re.search(r'^\|\s+[23]\s+\d+\s+',processes,re.M), 'User-assigned diagnostic GPUs are occupied'
job=next(x for x in json.loads((root/'active-training.json').read_bytes())['jobs'] if x['task']=='AppWorld')
assert job['pid']==proof['formal_driver_pid'] and job['observed_process_created_unix']==proof['formal_driver_birth']
env=os.environ.copy()
try:
 driver=psutil.Process(proof['formal_driver_pid']);assert driver.create_time()==proof['formal_driver_birth']
 raw=dict(x.split(b'=',1) for x in pathlib.Path('/proc',str(driver.pid),'environ').read_bytes().split(b'\0') if b'=' in x)
 env={k.decode():v.decode() for k,v in raw.items()}
 environment_source='Same live formal driver environment'
except psutil.NoSuchProcess:
 # Diagnostics remain possible after a formal task fails. Reuse its recorded
 # owner paths and provisioned environment; this never resubmits that task.
 env.update(DT_ROOT=job['dt_root'],DT_ENTRY_ROOT=job['entry'])
 environment_source='Provisioned metax-entry.env.sh plus recorded formal owner paths; prior formal driver is terminal'
for name in ['MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR']:env.pop(name,None)
env.update(CUDA_VISIBLE_DEVICES='2,3',VERL_ROOT=str(candidate),PADDING_DIAGNOSTIC_DIR=str(out),
 PADDING_REFERENCE_ROOT=str(base),VERL_RELEASE_UNUSED_HOST_CACHE='1')
assert env.get('VERL_TRIM_SHARED_PADDING')=='1', 'Use the already enabled formal trim branch'
env['PYTHONPATH']=':'.join([str(out),proof['entry'],str(candidate),env.get('PYTHONPATH','')])
argv=[env['VENV_PYTHON'],'-u',str(out/'verify_actor_right_padding.py')]
with (out/'diagnostic.log').open('wb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,started_unix=time.time(),created_unix=psutil.Process(p.pid).create_time(),argv=argv,gpus=[2,3],
 code_commit='@COMMIT@',launcher_sha256='@SHA@',prepared_source_commit=proof['code_commit'],environment_source=environment_source,
 scope='Same-weight owner padding assertion, literal inputs and one full B8/32768 native capacity update; formal jobs unchanged')
(out/'job.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'resources.log').open('wb') as log:
 observer=subprocess.Popen([env['VENV_PYTHON'],'@ENTRY@/observe_entry_resources.py','--job',str(out/'job.json'),
  '--output',str(out/'physical-resources.jsonl')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'observer.json').write_text(json.dumps(dict(pid=observer.pid,created_unix=psutil.Process(observer.pid).create_time()),indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT).replace('@OUT@', out)
       .replace('@COMMIT@', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
       .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
