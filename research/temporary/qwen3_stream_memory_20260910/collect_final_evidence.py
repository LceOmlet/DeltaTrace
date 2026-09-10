"""Freeze each finished study without modifying any measured row or source."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--phase',choices=['memory_confirmation','memory_author','rollout'],required=True);a=p.parse_args();root=a.root;phase=a.phase
names={'memory_confirmation':['benchmark_memory_confirmation.py','memory_confirmation_protocol.json','run_memory_confirmation.py','memory_confirmation_controller.log'],
 'memory_author':['check_memory_author.py','check_memory_contract.py','memory_author_plan.json','memory_author.log','check_graph_validation_probes.py'],
 'rollout':['benchmark_rollout.py','rollout_protocol.json','run_rollout.py','rollout_controller.log','postflight_memory.json','final_memory_mx_smi.txt']}[phase]
status=root/phase/('results.json' if phase=='memory_author' else 'queue.json')
assert json.loads(status.read_bytes())['status']=='complete'
selected=[root/n for n in names]
selected.extend(f for f in (root/phase).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
selected.append(Path(__file__))
sha=lambda b:hashlib.sha256(b).hexdigest();members={};total=0
for path in sorted(set(selected)):
 data=path.read_bytes();name=path.relative_to(root).as_posix();assert len(data)<50_000_000,name
 total+=len(data);members[name]={'sha256':sha(data),'bytes':len(data)}
assert total<150_000_000,total
manifest={'phase':phase,'files_count':len(members),'uncompressed_bytes':total,'files':members}
target=root/('evidence_'+phase+'.zip');assert not target.exists()
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for name in members:z.write(root/name,name)
 z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(target),'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'files':len(members),'uncompressed_bytes':total}))
