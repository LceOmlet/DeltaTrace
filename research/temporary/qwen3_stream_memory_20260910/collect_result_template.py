from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['result_template','result_profile'],required=True);a=p.parse_args()
r=Path(__file__).resolve().parent;s=lambda b:hashlib.sha256(b).hexdigest();folder=r/a.phase
state=json.loads((folder/'results.json').read_bytes());assert state['status'] in ('complete','failed')
names=['result_template_v1.py','result_template_graph.py','benchmark_'+a.phase+'.py',a.phase+'_protocol.json',a.phase+'.log','collect_result_template.py']
files=[r/n for n in names]+[f for f in folder.rglob('*') if f.is_file()]
members={f.relative_to(r).as_posix():{'bytes':f.stat().st_size,'sha256':s(f.read_bytes())} for f in sorted(set(files))}
out=r/('evidence_'+a.phase+'.zip');assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for n,v in members.items():
        data=(r/n).read_bytes();assert s(data)==v['sha256'];z.writestr(n,data)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(members),'files':members,'phase':a.phase,'phase_status':state['status']},indent=2)+'\n')
h=s(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
