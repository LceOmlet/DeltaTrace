"""Freeze a finished v3 phase without changing any measured source or result."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['confirmation','author','rollout'],required=True);args=p.parse_args()
root=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();phase='memory_v4_'+args.phase;folder=root/phase
state=json.loads((folder/('results.json' if args.phase=='author' else 'queue.json')).read_bytes())
assert state['status'] in ('complete','failed')
names=['collect_memory_v4.py','cpu_staged_model_loading.py']
if args.phase=='confirmation':
    names+=['benchmark_memory_v4.py','memory_v4_confirmation_protocol.json','run_memory_v4_confirmation.py',
            'memory_v4_confirmation_controller.log','summarize_memory_v4_confirmation.py','rollout_termination.json']
elif args.phase=='author':
    names+=['check_memory_v4_author.py','memory_v4_author_plan.json','memory_v4_author.log','check_memory_contract.py','check_memory_v4_graph_probes.py']
else:
    names+=['benchmark_memory_v4_rollout.py','memory_v4_rollout_protocol.json','run_memory_v4_rollout.py','memory_v4_rollout_controller.log']
selected=[root/n for n in names]
selected.extend(f for f in folder.rglob('*') if f.is_file())
for n in ['memory_v4_confirmation_summary.json','memory_v4_confirmation_summary.csv','run_memory_v4_followup.py']:
    if (root/n).exists():selected.append(root/n)
if args.phase=='confirmation':
    selected.extend(f for f in (root/'memory_production_v4_release').rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc')
members={};total=0
for f in sorted(set(selected)):
    data=f.read_bytes();assert len(data)<150_000_000;total+=len(data)
    members[f.relative_to(root).as_posix()]={'sha256':sha(data),'bytes':len(data)}
assert total<800_000_000
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members,'phase':phase,'phase_status':state['status']}
path=root/('evidence_'+phase+'.zip');assert not path.exists()
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for name,r in members.items():
        data=(root/name).read_bytes();assert sha(data)==r['sha256'],name;z.writestr(name,data)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
h=sha(path.read_bytes());print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
