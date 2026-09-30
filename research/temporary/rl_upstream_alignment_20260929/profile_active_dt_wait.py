"""Short read-only samples of the two current AppWorld workers; no job changes."""
from stage_environment_entry import remote, ROOT, ENTRY

remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json, psutil, subprocess, time
root=Path('@ROOT@')
j=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
p=psutil.Process(j['pid'])
assert abs(p.create_time()-j['observed_process_created_unix']) < .02
workers=[p for p in p.children(recursive=True) if 'WorkerDict' in p.name()]
assert len(workers)==2
out=root/'receipts/owner-b8-dispatch-20260930/dt-wait-profile'
out.mkdir(exist_ok=True)
def sample(p):
    dest=out/f'{p.pid}-{int(time.time())}.json'
    r=subprocess.run(['/opt/conda/bin/py-spy','record','--pid',str(p.pid),'--duration','12',
        '--rate','40','--idle','--format','speedscope','--output',str(dest)],capture_output=True,text=True,timeout=25)
    if r.returncode: return dict(pid=p.pid,error=r.stderr)
    data=json.loads(dest.read_text());frames=data['shared']['frames'];counts=Counter();total=0
    for profile in data['profiles']:
        if 'samples' not in profile:continue
        for stack,weight in zip(profile['samples'],profile.get('weights',[1]*len(profile['samples']))):
            names=[frames[i]['name'] for i in stack]
            if not any('compute_dt_token_advantages' in n for n in names):continue
            total+=weight
            counts[' / '.join(names[-5:])]+=weight
    return dict(pid=p.pid,path=str(dest),sample_weight=total,top=counts.most_common(8))
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(sample,workers))
(out/'summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
