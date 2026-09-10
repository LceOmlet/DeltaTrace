"""Freeze both complete calls, including raw vectors and all setup costs."""
from pathlib import Path
import hashlib,json,zipfile
A=Path(__file__).resolve().parent;s=lambda b:hashlib.sha256(b).hexdigest()
q=json.loads((A/'fa_mma_pilot/queue.json').read_bytes());assert q['status'] in ('complete','failed')
names=['collect_fa_mma_pilot.py','benchmark_fa_mma_pilot_short.py','fa_mma_pilot_short_protocol.json','benchmark_fa_mma_pilot_rollout.py','fa_mma_pilot_rollout_protocol.json','cpu_staged_model_loading.py','run_fa_mma_pilot.py','fa_mma_pilot_controller.log']
paths=[A/n for n in names]+[f for f in (A/'fa_mma_pilot').rglob('*') if f.is_file()]
out=A/'evidence_fa_mma_pilot.zip';assert not out.exists();members={}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(set(paths)):
        n=f.relative_to(A).as_posix();data=f.read_bytes();members[n]={'bytes':len(data),'sha256':s(data)};z.writestr(n,data)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(members),'files':members,'phase':'fa_mma_pilot','phase_status':q['status']},indent=2)+'\n')
h=s(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
