from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fa_owner64','fa_owner64_split'],required=True);a=p.parse_args();root=Path(__file__).resolve().parent;folder=root/a.phase
s=lambda b:hashlib.sha256(b).hexdigest();state=json.loads((folder/('controller_v2.json' if a.phase=='fa_owner64' else 'controller.json')).read_bytes());assert state['status'] in ('complete','failed')
members={};paths=[Path(__file__)]+[f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='review_bundle.zip'];out=root/('evidence_'+a.phase+'.zip');assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in paths:
        n=f.relative_to(root).as_posix();data=f.read_bytes();members[n]={'bytes':len(data),'sha256':s(data)};z.writestr(n,data)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(members),'files':members,'phase':a.phase,'phase_status':state['status']},indent=2)+'\n')
h=s(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
