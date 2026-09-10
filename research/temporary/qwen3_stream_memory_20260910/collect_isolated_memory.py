"""Freeze all twelve capacity cells and the zero-call FT setup interruption."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent
assert json.loads((root/'isolated_memory/queue.json').read_bytes())['status']=='complete'
names=['benchmark_isolated_memory.py','isolated_memory_protocol.json','run_isolated_memory.py','run_isolated_memory_resume.py','isolated_memory_controller.log','isolated_memory_resume_controller.log','collect_isolated_memory.py']
selected=[root/n for n in names]+[f for f in (root/'isolated_memory').rglob('*') if f.is_file()]
sha=lambda b:hashlib.sha256(b).hexdigest();members={};total=0
for f in sorted(set(selected)):
 data=f.read_bytes();assert len(data)<50_000_000;total+=len(data);members[f.relative_to(root).as_posix()]={'sha256':sha(data),'bytes':len(data)}
assert total<150_000_000
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members}
path=root/'evidence_isolated_memory.zip';assert not path.exists()
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for name in members:z.write(root/name,name)
 z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256_parts':[sha(path.read_bytes())[:32],sha(path.read_bytes())[32:]],'files':len(members)}))
