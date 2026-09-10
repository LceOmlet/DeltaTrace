"""Archive failed rollout cells, exactness diagnostics, and repair pilot verbatim."""
from pathlib import Path
import hashlib,json,zipfile

root=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
repair=json.loads((root/'rounding_repair/queue.json').read_bytes())
assert repair['status'] in ('complete','failed')
# The initial rollout controller must remain stopped or be terminated before
# taking this snapshot. Its last running marker is preserved verbatim.
pid=Path('/proc/308461/status')
if pid.exists():assert pid.read_text().split('State:')[1].splitlines()[0].strip().startswith(('T','Z'))
for folder in ['rollout_diagnostic','rollout_boundaries']:
    assert json.loads((root/folder/'results.json').read_bytes())['status']=='complete'
names=['benchmark_rollout.py','rollout_protocol.json','run_rollout.py','rollout_pause.json',
 'diagnose_rollout_exactness.py','rollout_diagnostic_protocol.json',
 'diagnose_rollout_boundaries.py','rollout_boundaries_protocol.json',
 'diagnostic_boundary_ops.py','diagnostic_boundary_finite.py','diagnostic_boundary_graph.py',
 'rounding_repaired_finite.py','rounding_repaired_core.py','rounding_repaired_graph.py',
 'benchmark_rounding_short.py','benchmark_rounding_rollout.py',
 'rounding_short_protocol.json','rounding_rollout_protocol.json','run_rounding_repair.py',
 'rounding_repair_controller.log','collect_rounding_evidence.py']
selected=[root/n for n in names]
for folder in ['rollout','rollout_diagnostic','rollout_boundaries','rounding_repair']:
    selected.extend(p for p in (root/folder).rglob('*') if p.is_file())
selected.extend(p for p in root.glob('*diagnostic*.log') if p.is_file())
selected.extend(p for p in root.glob('*boundaries*.log') if p.is_file())
members={};total=0
for path in sorted(set(selected)):
    data=path.read_bytes();assert len(data)<100_000_000
    total+=len(data);members[path.relative_to(root).as_posix()]={'sha256':sha(data),'bytes':len(data)}
assert total<300_000_000
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members,
 'scope':'Initial rollout partial/failed cells retained without editing; same-input diagnosis and repaired pilot. No failed candidate is a valid final curve point.',
 'repair_status':repair['status']}
path=root/'evidence_rounding.zip';assert not path.exists()
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for name,record in members.items():
        data=(root/name).read_bytes();assert sha(data)==record['sha256'],name;z.writestr(name,data)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
digest=sha(path.read_bytes())
print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256_parts':[digest[:32],digest[32:]],'files':len(members)}))
