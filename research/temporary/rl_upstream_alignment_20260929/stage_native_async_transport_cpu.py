"""Prepare a separate async transport candidate and run bounded CPU checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from stage_environment_entry import remote,ROOT,ENTRY,AUDIT,REPO,SCP,SSH


def main():
    base=ROOT+'/candidates/appworld-native-async-015-transport-20261002'
    bundle=AUDIT/'appworld-official-async-20261002/transport-source.tar'
    files=['patch_verl_async_vllm_015.py','patch_verl_environment_entry.py']
    candidate=AUDIT/'appworld-official-async-20261002/transport-candidate-entry'
    with tarfile.open(bundle,'w') as archive:
        for name in files:archive.add(REPO/'experiments/rl'/name,arcname=name)
        for name in ['loop_async_transport.py','loop_owner_rollout.py','owner_rollout_scope.py']:
            archive.add(candidate/name,arcname=name)
        for name in ['verify_async_owner_015_interfaces.py','probe_native_async_transport_cpu.py']:
            archive.add(AUDIT/name,arcname=name)
    remote(f'mkdir -p {base}/overlay\n')
    subprocess.run(SCP+[str(bundle),f'{SSH[-1]}:{base}/overlay/source.tar'],check=True)
    remote(fr'''source {ENTRY}/metax-entry.env.sh
tar -xf {base}/overlay/source.tar -C {base}/overlay
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,pathlib,shutil,subprocess,sys,time
root=pathlib.Path('{ROOT}');base=pathlib.Path('{base}');overlay=base/'overlay'
active=json.loads((root/'active-training.json').read_text())
job=next(j for j in active['jobs'] if j['task']=='AppWorld')
owner=base/'verl';entry=base/'entry'
assert not owner.exists() and not entry.exists(),'Candidate already exists; preserve its source/receipt'
shutil.copytree(job['verl_root'],owner,ignore=shutil.ignore_patterns('.git','__pycache__'))
shutil.copytree(job['entry'],entry,ignore=shutil.ignore_patterns('__pycache__'))
sys.path.insert(0,str(overlay))
from patch_verl_async_vllm_015 import apply
names=['verl/workers/rollout/vllm_rollout/vllm_async_server.py',
       'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
       'verl/workers/sharding_manager/fsdp_vllm.py','verl/trainer/ppo/ray_trainer.py']
sources={{}};originals=base/'originals'
for name in names:
 p=owner/name;q=originals/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 sources[name]={{'before_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
apply(owner)
for name in names:sources[name]['after_sha256']=hashlib.sha256((owner/name).read_bytes()).hexdigest()
for name in ['loop_async_transport.py','loop_owner_rollout.py','owner_rollout_scope.py']:
 shutil.copy2(overlay/name,entry/name)
 ast.parse((entry/name).read_text())
receipt=dict(prepared_unix=time.time(),status='unaccepted CPU candidate; not deployed or enabled',
 owner=str(owner),entry=str(entry),source_owner=job['verl_root'],source_entry=job['entry'],
 sources=sources,patch_sha256=hashlib.sha256((overlay/'patch_verl_async_vllm_015.py').read_bytes()).hexdigest())
(base/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
export PYTHONPATH={base}/entry:{base}/verl:$PYTHONPATH
"$VENV_PYTHON" {base}/overlay/verify_async_owner_015_interfaces.py --candidate {base}/verl --root {ROOT} --output {base}/probe-interface
"$VENV_PYTHON" {base}/overlay/probe_native_async_transport_cpu.py --candidate {base}/verl --entry {base}/entry --original-spmd {base}/originals/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py --output {base}/probe-transport/transport.json
''')
    target=AUDIT/'appworld-official-async-20261002/transport-cpu-probe'
    target.mkdir(parents=True,exist_ok=True)
    for name in ['prepared.json','probe-interface/interfaces.json','probe-transport/transport.json']:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(SCP+[f'{SSH[-1]}:{base}/{name}',str(destination)],check=True)


if __name__=='__main__':main()
