"""Freeze the final full-file identity audit after the benchmark queue exits."""
from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parent;s=lambda b:hashlib.sha256(b).hexdigest()
d=json.loads((P/'postflight_memory_v4.json').read_bytes());assert d['status'] in ('verified','failed')
names=['collect_memory_v4_postflight.py','postflight_memory_v4.py','postflight_memory_v4_plan.json','postflight_memory_v4.json','final_memory_v4_mx_smi.txt','memory_v4_followup.json']
files={n:(P/n).read_bytes() for n in names if (P/n).exists()}
out=P/'evidence_memory_v4_postflight.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in files.items():z.writestr(n,b)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(files),'phase':'memory_v4_postflight','phase_status':d['status'],'files':{n:{'bytes':len(b),'sha256':s(b)} for n,b in files.items()}},indent=2)+'\n')
h=s(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(files)}))
