"""Freeze the tested transport seam over the actual AppWorld deployment.

Preparation only. The existing submission helper and original VERL loader own
launch/restore; no live process, manifest, task setting or numerical code changes.
"""
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,psutil,shutil,time,xml.etree.ElementTree as ET
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tested=root/'receipts/owner-b8-dispatch-20260930/appworld-rank-completion'
proof=read(tested/'final-cpu-tests.json')
assert proof['exit_code']==0 and proof['ray_resources']['num_gpus']==0
xml=ET.parse(tested/'final-cpu-tests.xml').getroot()
assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in xml.iter('testsuite'))==0
for name in ('final-cpu-tests.log','final-cpu-tests.xml'):
    assert sha(tested/name)==proof['receipts'][name]['sha256'],name
candidate=tested/'appworld-rank-completion-20261002/loop_owner_rollout.py'
assert sha(candidate)==proof['sources']['candidate']['sha256']
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='AppWorld')
process=psutil.Process(job['pid'])
assert abs(process.create_time()-job['observed_process_created_unix'])<.02
source=read(Path(job['output'])/'source.json')
prior_path=Path(source['prepared_receipt']);prior=read(prior_path)
old_entry=Path(job['entry']);owner=Path(job['verl_root'])
for name,expected in source['entry_sha256'].items():assert sha(old_entry/name)==expected,name
for name,expected in source['verl_sha256'].items():assert sha(owner/name)==expected,name
for name,expected in prior['owner_head_sha256'].items():assert sha(owner/name)==expected,name
for name,expected in prior['dt_source_sha256'].items():assert sha(Path(job['dt_root'])/name)==expected,name
assert sha(old_entry/'loop_owner_rollout.py')==proof['sources']['original']['sha256']
base=root/'candidates/appworld-rank-completion-20261002'
receipt=tested/'release'
assert not base.exists() and not receipt.exists(), 'Inspect immutable existing preparation'
entry=base/'entry';receipt.mkdir(parents=True)
shutil.copytree(old_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(candidate,entry/'loop_owner_rollout.py')
actual={name:sha(entry/name) for name in source['entry_sha256']}
assert [name for name,h in actual.items() if h!=source['entry_sha256'][name]]==['loop_owner_rollout.py']
record=dict(prior,prepared_unix=time.time(),status='prepared_not_launched',
    entry=str(entry),verl_root=str(owner),prior_driver_pid=job['pid'],
    prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    future_checkpoint_root=job['checkpoints'],entry_sha256=actual,
    prior_prepared_receipt=str(prior_path),prior_prepared_receipt_sha256=sha(prior_path),
    preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
    completion_transport_code_commit=proof['test_code_commit'],
    completion_transport_receipt=str(tested/'final-cpu-tests.json'),
    completion_transport_receipt_sha256=sha(tested/'final-cpu-tests.json'),
    completion_transport_sources=proof['sources'],
    scope='Native Ray ActorPool completion order and original per-worker VERL RPC only; all task, actor, DT, loss, workload and scope sources preserved. Not launched.')
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','entry','verl_root','prior_driver_pid','future_checkpoint_root']},indent=2))
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
   .replace('@REVISION@', repr(revision)).replace('@SCRIPT_SHA@', repr(script_sha)))
target=AUDIT/'appworld-rank-completion-20261002'
subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/appworld-rank-completion/release/prepared.json',
                   str(target/'prepared.json')],check=True)
