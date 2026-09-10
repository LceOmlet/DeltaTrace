from pathlib import Path
import hashlib,json,zipfile
p=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
assert json.loads((p/'headroom/results.json').read_bytes())['status']=='complete'
protocol=json.loads((p/'headroom_protocol.json').read_bytes())
selected=[p/n for n in [*protocol['repair_modules'],'benchmark_headroom_short.py','headroom_protocol.json','headroom.log','collect_headroom_evidence.py']]
selected.extend(f for f in (p/'headroom').rglob('*') if f.is_file())
files={}
for f in sorted(set(selected)):
    b=f.read_bytes();files[f.relative_to(p).as_posix()]={'sha256':sha(b),'bytes':len(b)}
manifest={'files_count':len(files),'uncompressed_bytes':sum(r['bytes'] for r in files.values()),'files':files}
out=p/'evidence_headroom.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for n,r in files.items():
        b=(p/n).read_bytes();assert sha(b)==r['sha256'];z.writestr(n,b)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
h=sha(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(files)}))
