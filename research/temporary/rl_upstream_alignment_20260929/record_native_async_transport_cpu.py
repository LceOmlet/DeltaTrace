"""Attach prepared-only receipts without refreshing or changing active jobs."""
import hashlib
import json
from pathlib import Path

from stage_environment_entry import AUDIT,REPO,ROOT

target=AUDIT/'appworld-official-async-20261002/transport-cpu-probe'
prepared=json.loads((target/'prepared.json').read_text())
interface=json.loads((target/'probe-interface/interfaces.json').read_text())
transport=json.loads((target/'probe-transport/transport.json').read_text())
assert interface['passed'] and transport['passed']
snapshot=REPO/'experiments/rl/current_runtime.json'
runtime=json.loads(snapshot.read_text())
identity='appworld-native-async-015-transport-20261002'
record=dict(id=identity,status='unaccepted, prepared-only; CPU interfaces verified; no engine or formal deployment',
    prepared_unix=prepared['prepared_unix'],observed_probe_finished_unix=interface['finished_unix'],
    entry=prepared['entry'],verl_root=prepared['owner'],
    dt_root=ROOT+'/releases/c9cd147',active_jobs_with_this_entry=[],configuration_changes={},
    preparation_repository_commit='e3230713478a10434f30dbd06607e01df6a8d7d1',
    patched_owner_files=prepared['sources'],patch_sha256=prepared['patch_sha256'],
    receipt=dict(path=ROOT+'/candidates/'+identity+'/prepared.json',exists=True,
        sha256=hashlib.sha256((target/'prepared.json').read_bytes()).hexdigest()),
    verification_receipts=[],
    scope='Native API bindings and exact transport conversion only; no model numerics, LoRA sync runtime, capacity or speedup claim')
for name in ['probe-interface/interfaces.json','probe-transport/transport.json']:
    record['verification_receipts'].append(dict(
        path=ROOT+'/candidates/'+identity+'/'+name,
        local=str((target/name).relative_to(REPO)).replace('\\','/'),
        sha256=hashlib.sha256((target/name).read_bytes()).hexdigest()))
versions=runtime['prepared_versions']
assert all(v['id']!=identity for v in versions),'Preserve existing prepared record instead of overwriting it'
versions.append(record)
snapshot.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(id=identity,source_snapshot_observed_unix_unchanged=runtime['observed_unix'],
    active_jobs_unchanged=[j['pid'] for j in runtime['jobs']],status=record['status']),ensure_ascii=False))
