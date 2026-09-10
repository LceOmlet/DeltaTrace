"""Preserve failed bounded seeds, both diagnostics, and exact-seed successor."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
for name in ['bounded_target','exact_target']:
    assert json.loads((root/name/'queue.json').read_bytes())['status'] in ('complete','failed')
for name in ['target_diagnostic','bounded_diagnostic']:
    assert json.loads((root/name/'results.json').read_bytes())['status']=='complete'
names=['bounded_target_derivation.json','exact_target_derivation.json']
# Derivation metadata are local producers; if not uploaded, source files and
# frozen remote protocols remain the complete actual execution provenance.
selected=[]
for pattern in ['bounded_target_*.py','bounded_target_*protocol.json','benchmark_bounded_target_*.py','run_bounded_target.py',
                'exact_target_*.py','exact_target_*protocol.json','benchmark_exact_target_*.py','run_exact_target.py',
                'bounded_diagnostic*.py','bounded_diagnostic_protocol.json','diagnose_bounded_program.py',
                'diagnose_target_rows.py','target_diagnostic_protocol.json','*target*controller.log',
                'target_diagnostic.log','bounded_diagnostic.log','collect_target_evidence.py']:
    selected.extend(p for p in root.glob(pattern) if p.is_file())
for folder in ['bounded_target','target_diagnostic','bounded_diagnostic','exact_target']:
    selected.extend(p for p in (root/folder).rglob('*') if p.is_file())
members={};total=0
for p in sorted(set(selected)):
    data=p.read_bytes();assert len(data)<100_000_000;total+=len(data)
    members[p.relative_to(root).as_posix()]={'sha256':sha(data),'bytes':len(data)}
assert total<300_000_000
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members,
          'scope':'All bounded-target failures and exact-seed successor, including counterexample to isolated target-operation success.'}
path=root/'evidence_target.zip';assert not path.exists()
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for n,r in members.items():
        data=(root/n).read_bytes();assert sha(data)==r['sha256'];z.writestr(n,data)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
h=sha(path.read_bytes());print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
