"""Archive bounded evidence only, with content hashes for every member."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['benchmark','author'],required=True);args=p.parse_args();root=args.root
sha=lambda b:hashlib.sha256(b).hexdigest()
selected=[]
if args.phase=='benchmark':
    assert json.loads((root/'confirmation_v2/queue.json').read_bytes())['status']=='complete'
    selected.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.json','.log'])
    for folder in ['pilot_v1','pilot_v2','pilot_v3','pilot_v4','compact_pilot','compact_v2','confirmation','confirmation_v2']:
        selected.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.npz','.log'])
    for folder,protocol in [('stream_production_release','confirmation_protocol.json'),('compact_production_release','confirmation_v2_protocol.json')]:
        names=list(json.loads((root/protocol).read_bytes())['runtime_files'])+['environment.json','official_exp1/run_time_curve.py']
        selected.extend(root/folder/n for n in names)
else:
    assert json.loads((root/'author/results.json').read_bytes())['status']=='complete'
    assert json.loads((root/'postflight.json').read_bytes())['status']=='verified'
    selected.extend(root/n for n in ['check_author.py','author_plan.json','check_stream_storage.py','check_graph_validation_probes.py','author.log','postflight_verify.py','postflight_plan.json','postflight.json','postflight.log','final_mx_smi.txt'])
    selected.extend(p for p in (root/'author').rglob('*') if p.is_file())
members={};total=0
for path in sorted(set(selected)):
    data=path.read_bytes();name=path.relative_to(root).as_posix();assert path.is_file() and len(data)<40_000_000,name
    total+=len(data);members[name]={'sha256':sha(data),'bytes':len(data)}
assert total<100_000_000,total
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members}
target=root/('evidence_'+args.phase+'.zip');assert not target.exists()
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(root/name,name)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(target),'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'files':len(members),'uncompressed_bytes':total}))
