"""Freeze the CPU-tested eval IPC repair; original submitter owns resumption.

Only three current entry files change. Actor, DT, LOOP, workload and native
checkpoint loading remain the already recorded owners. Does not submit jobs.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_appworld_eval_client_routing_20261005 import EXPECTED, patched_sources
from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


def main():
    baseline = AUDIT / 'appworld-eval-client-routing-20261005/baseline'
    files = patched_sources({n: (baseline / n).read_bytes() for n in EXPECTED})
    proof = REPO / 'experiments/rl/results_appworld_eval_client_routing_cpu_20261005.json'
    evidence = json.loads(proof.read_bytes())
    assert evidence['test']['passed'] == 10 and evidence['test']['failed'] == 0
    for n, content in files.items():
        assert hashlib.sha256(content).hexdigest() == evidence['files'][n]['candidate_sha256']
    payload = dict(expected=EXPECTED, files={n: base64.b64encode(b).decode() for n, b in files.items()},
        proof=json.loads(proof.read_bytes()), proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1
"$VENV_PYTHON" - <<'PY'
import ast,base64,hashlib,json,os,runpy,shutil,sys,time
from pathlib import Path
root=Path(@ROOT@);payload=json.loads(@PAYLOAD@)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='AppWorld')
assert job['pid']==3658900 and job['devices']==[2,3]
prior_path=root/'candidates/appworld-native-prefix-resume-20261005-v2/prepared.json'
prior=read(prior_path);old=Path(job['entry'])
assert prior['entry']==str(old) and prior['verl_root']==job['verl_root'] and prior['dt_root']==job['dt_root']
for n,h in prior['entry_sha256'].items():assert sha(old/n)==h,n
for n,h in prior['owner_sha256'].items():assert sha(Path(job['verl_root'])/n)==h,n
for n,h in prior['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/n)==h,n
marker=Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
assert int(marker.read_text())==20
checkpoint=marker.parent/'global_step_20'
assert all(p.is_file() and p.stat().st_size for p in [checkpoint/'data.pt']+[
    checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
    for rank in range(2) for kind in ('model','optim','extra_state')])
base=root/'candidates/appworld-eval-client-routing-resume-20261005-v1'
assert not base.exists(), 'Preserve immutable candidates; inspect any existing attempt'
entry=base/'entry';base.mkdir();shutil.copytree(old,entry,symlinks=True)
def tree(p):
    return {str(f.relative_to(p)):os.readlink(f) if f.is_symlink() else sha(f)
            for f in p.rglob('*') if f.is_file() or f.is_symlink()}
before=tree(old);assert tree(entry)==before
for n,b in payload['files'].items():
    assert sha(entry/n)==payload['expected'][n]
    content=base64.b64decode(b);ast.parse(content);(entry/n).write_bytes(content)
after=tree(entry)
assert {n for n in before.keys()|after.keys() if before.get(n)!=after.get(n)}==set(payload['files'])
os.environ.update(DT_ENTRY_ROOT=str(entry),VERL_ROOT=job['verl_root'],DT_ROOT=job['dt_root'],
    LOOP_ROOT=prior['loop_root'],APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
sys.path[:0]=[str(entry),job['verl_root'],prior['loop_root'],str(Path(job['dt_root'])/'experiments/rl'),job['dt_root']]
a=runpy.run_path(str(old/'launch_appworld_native.py'))['options_for'](Path('/same-output'),resume_from=checkpoint)
b=runpy.run_path(str(entry/'launch_appworld_native.py'))['options_for'](Path('/same-output'),resume_from=checkpoint)
normalized=dict(b[0]);normalized['data.custom_cls.path']=a[0]['data.custom_cls.path']
assert normalized==a[0] and b[1]==a[1], 'Only entry location may change'
(base/'prior-prepared.json').write_bytes(prior_path.read_bytes())
(base/'cpu-dispatch-verification.json').write_text(json.dumps(payload['proof'],indent=2)+'\n')
record=dict(prior,prepared_unix=time.time(),status='prepared_only_eval_client_routing_cpu_verified_not_submitted',
    scope='Keep LOOP selected eval client index across native IPC only; training default, numerical cores and official task options unchanged',
    entry=str(entry),entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},
    prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    future_checkpoint_root=job['checkpoints'],resume_checkpoint=str(checkpoint),
    preparation_repository_commit=payload['revision'],preparation_script_sha256=payload['helper_sha256'],
    evaluation_routing_verification=str(base/'cpu-dispatch-verification.json'),
    evaluation_routing_verification_sha256=sha(base/'cpu-dispatch-verification.json'),
    evaluation_routing_local_receipt_sha256=payload['proof_sha256'],
    unchanged_formal_options=b[0],unchanged_sampling=b[1],unchanged_actor_and_dt=True)
(base/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(prepared=str(base/'prepared.json'),entry=str(entry),entry_changes=sorted(payload['files']),
    checkpoint=str(checkpoint),status=record['status'])),flush=True)
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', repr(ROOT)).replace('@PAYLOAD@', repr(json.dumps(payload)))
    remote(script)
    local = AUDIT / 'appworld-eval-client-routing-20261005/remote-prepared.json'
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/candidates/appworld-eval-client-routing-resume-20261005-v1/prepared.json',str(local)],check=True)


if __name__ == '__main__':
    main()
