"""Freeze the reproduced reward/mapping repair and existing host-cache override.

Preparation only. Original AgentGym and VERL still own truncation, rewards,
sampling, updates and checkpoints. Does not start or stop a training process.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_sql_native_host_cache_20261005 import build_payload as accepted_host_payload
from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


NAMES = ('textcraft_owner_rollout.py', 'owner_trajectory_batch.py', 'dt_training_batch.py')


def main():
    baseline = AUDIT/'textcraft-late-sampling-audit-20261005/baseline'
    bindings = json.loads((baseline/'source.json').read_bytes())
    host = accepted_host_payload()
    payload = dict(host=host, before={n: bindings['files'][n]['sha256'] for n in NAMES},
        files={n:base64.b64encode((REPO/'experiments/rl'/n).read_bytes()).decode() for n in NAMES},
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VENV_PYTHON" - <<'PY'
import ast,base64,hashlib,json,os,runpy,shutil,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(@ROOT@);payload=json.loads(@PAYLOAD@)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='TextCraft')
assert job['pid']==3218909 and job['devices']==[4,5]
prior_path=root/'receipts/owner-b8-dispatch-20260930/textcraft-rollout-scope/prepared.json'
prior=read(prior_path);old=Path(job['entry']);old_owner=Path(job['verl_root'])
assert prior['entry']==str(old) and prior['verl_root']==str(old_owner)
for n,h in prior['entry_sha256'].items():assert sha(old/n)==h,n
for n,h in prior['owner_sha256'].items():assert sha(old_owner/n)==h,n
for n,h in prior['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/n)==h,n
proof=root/'receipts/owner-b8-dispatch-20260930/textcraft-truncated-terminal-20261005'
xml=proof/'cpu-tests.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==18
assert sum(int(s.get('skipped','0')) for s in suites)==1
assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors'))
for n,b in payload['files'].items():assert (proof/n).read_bytes()==base64.b64decode(b)
marker=Path(job['checkpoints'])/'latest_checkpointed_iteration.txt';step=int(marker.read_text())
assert step==100
checkpoint=marker.parent/f'global_step_{step}'
assert all(p.is_file() and p.stat().st_size for p in [checkpoint/'data.pt']+[
 checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
 for rank in range(2) for kind in ('model','optim','extra_state')])
base=root/'candidates/textcraft-truncated-terminal-resume-20261005-v1'
assert not base.exists(), 'Preserve immutable candidates; inspect existing attempt'
entry=base/'entry';owner=base/'verl';base.mkdir()
shutil.copytree(old,entry,symlinks=True);shutil.copytree(old_owner,owner,symlinks=True)
def tree(p):
 return {str(f.relative_to(p)):os.readlink(f) if f.is_symlink() else sha(f)
         for f in p.rglob('*') if f.is_file() or f.is_symlink()}
before=tree(old);owner_before=tree(old_owner)
assert tree(entry)==before and tree(owner)==owner_before
for n,b in payload['files'].items():
 assert sha(entry/n)==payload['before'][n]
 data=base64.b64decode(b);ast.parse(data);(entry/n).write_bytes(data)
worker=owner/'verl/workers/fsdp_workers.py'
assert worker.read_bytes()==base64.b64decode(payload['host']['original_worker'])
# Reuse the accepted resource patch function, not a second implementation.
namespace={'__name__':'_accepted_host_patch'}
exec(compile(base64.b64decode(payload['host']['patch_source']),'accepted_host_patch.py','exec'),namespace)
patched=namespace['patch_source'](worker.read_text()).encode()
assert patched==base64.b64decode(payload['host']['patched_worker']);worker.write_bytes(patched)
launcher=entry/'launch_textcraft_native.py';data=launcher.read_bytes()
eol=b'\r\n' if b'\r\n' in data else b'\n'
anchor=b'    options = {**runtime_options(),'+eol
assert data.count(anchor)==1
offset=data.index(anchor)+len(anchor)
addition=b"        '+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE': '1',"+eol
launcher.write_bytes(data[:offset]+addition+data[offset:]);ast.parse(launcher.read_bytes())
after=tree(entry);owner_after=tree(owner)
assert {n for n in before.keys()|after.keys() if before.get(n)!=after.get(n)}==set(payload['files'])|{'launch_textcraft_native.py'}
assert {n for n in owner_before.keys()|owner_after.keys() if owner_before.get(n)!=owner_after.get(n)}=={'verl/workers/fsdp_workers.py'}
launch=read(Path(job['output'])/'launch.json');agentgym=launch['options']['+env.textcraft']['owner_root']
os.environ.update(AGENTGYM_RL_ROOT=agentgym,VERL_ROOT=str(owner),DT_ROOT=job['dt_root'],DT_ENTRY_ROOT=str(old))
sys.path[:0]=[str(entry),str(owner),str(Path(job['dt_root'])/'experiments/rl'),job['dt_root']]
data_root=Path(job['argv'][job['argv'].index('--data')+1]);same_output=Path('/same-output')
a=runpy.run_path(str(old/'launch_textcraft_native.py'))['options_for'](data_root,same_output,resume_from=checkpoint)
os.environ['DT_ENTRY_ROOT']=str(entry)
b=runpy.run_path(str(launcher))['options_for'](data_root,same_output,resume_from=checkpoint)
normalized=dict(b[0]);normalized['data.custom_cls.path']=a[0]['data.custom_cls.path']
resource='+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE'
assert normalized.pop(resource)=='1' and normalized==a[0] and b[1]==a[1]
(base/'prior-prepared.json').write_bytes(prior_path.read_bytes())
record=dict(prior,prepared_unix=time.time(),status='prepared_only_truncated_terminal_reward_verified_not_submitted',
 entry=str(entry),verl_root=str(owner),prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
 future_checkpoint_root=job['checkpoints'],resume_checkpoint=str(checkpoint),
 entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
 owner_sha256={n:sha(owner/n) for n in prior['owner_sha256']},
 owner_head_sha256={n:sha(owner/n) for n in prior['owner_sha256']},
 preparation_repository_commit=payload['revision'],preparation_script_sha256=payload['helper_sha256'],
 reward_mask_comparison=str(xml),reward_mask_comparison_sha256=sha(xml),
 resource_environment={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'},
 resource_patch_commit='6781bdde60b4873f8336e12dd0e6c2ea1fca22b3',
 unchanged_formal_options=b[0],unchanged_sampling=b[1],
 scope='Actual executed rewards remain at their original events; existing native slices only select DT requests after complete returns. No event, token endpoint, Q/V/PPO or official task parameter changed. Fold existing completed host-cache runtime override into restart.')
(base/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(prepared=str(base/'prepared.json'),checkpoint=str(checkpoint),status=record['status'])),flush=True)
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',repr(ROOT)).replace('@PAYLOAD@',repr(json.dumps(payload)))
    remote(script)
    destination=AUDIT/'textcraft-late-sampling-audit-20261005/remote-prepared.json'
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/candidates/textcraft-truncated-terminal-resume-20261005-v1/prepared.json',str(destination)],check=True)


if __name__ == '__main__':
    main()
