"""Freeze all remaining cached-FA pilots and live template validation controls."""
from pathlib import Path
import hashlib,json,zipfile
A=Path(__file__).resolve().parent;s=lambda b:hashlib.sha256(b).hexdigest()
q=json.loads((A/'fa_cached_followup/queue.json').read_bytes());assert q['status'] in ('complete','failed')
names=['collect_fa_cached_followup.py','benchmark_fa_cached_short.py','fa_cached_short_protocol.json','benchmark_fa_cached_rollout.py','fa_cached_rollout_protocol.json',
       'check_template_graph_validation_probes.py','run_template_validation.py','run_fa_cached_followup.py','fa_cached_followup_plan.json','fa_cached_followup_controller.log','result_template_v1.py','result_template_graph.py']
paths=[A/n for n in names]
for n in ['fa_cached_followup','template_validation']:paths.extend(f for f in (A/n).rglob('*') if f.is_file())
out=A/'evidence_fa_cached_followup.zip';assert not out.exists();members={}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(set(paths)):
        n=f.relative_to(A).as_posix();data=f.read_bytes();members[n]={'bytes':len(data),'sha256':s(data)};z.writestr(n,data)
    z.writestr('archive_manifest.json',json.dumps({'files_count':len(members),'files':members,'phase':'fa_cached_followup','phase_status':q['status']},indent=2)+'\n')
h=s(out.read_bytes());print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256_parts':[h[:32],h[32:]],'files':len(members)}))
