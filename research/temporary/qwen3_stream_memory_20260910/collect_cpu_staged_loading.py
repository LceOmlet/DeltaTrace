from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent;phase='cpu_staged_loading';sha=lambda b:hashlib.sha256(b).hexdigest()
d=json.loads((root/phase/'results.json').read_bytes());assert d['status'] in ('complete','failed')
files=[root/'check_cpu_staged_loading.py',root/'cpu_staged_loading_plan.json',root/'cpu_staged_loading.log',Path(__file__)]+[p for p in (root/phase).rglob('*') if p.is_file()]
out=root/('evidence_'+phase+'.zip');assert not out.exists();members={}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:
        n=f.relative_to(root).as_posix();b=f.read_bytes();members[n]={'bytes':len(b),'sha256':sha(b)};z.writestr(n,b)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(members),'files':members,'phase':phase,'phase_status':d['status']},indent=2)+'\n')
h=sha(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
