"""Observe existing owner-loop counters without instrumenting training."""
from stage_environment_entry import remote,ROOT,ENTRY
remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,subprocess,re,time
root=Path('@ROOT@');out=root/'receipts/owner-b8-dispatch-20260930/phase-observations'
out.mkdir(exist_ok=True);observed=int(time.time())
jobs=json.loads((root/'active-training.json').read_text())['jobs']
for j in jobs:
    if j['task']=='AppWorld':continue
    children=psutil.Process(j['pid']).children(recursive=True)
    targets=[p for p in children if 'TaskRunner' in p.name() or 'WorkerDict' in p.name()][:2]
    for p in targets:
        result=subprocess.run(['/opt/conda/bin/py-spy','dump','--locals','-p',str(p.pid)],capture_output=True,text=True,timeout=20)
        receipt=out/f"{observed}-{j['task']}-{p.pid}-counts.txt"
        receipt.write_text(result.stdout)
        print('OBSERVATION',observed,'RECEIPT',receipt)
        print(j['task'],p.pid,p.name())
        active=False
        for line in result.stdout.splitlines():
            if re.match(r'^    [^ ]',line):
                active=any(n in line for n in ['_validate (','vanilla_multi_turn_loop (','compute_log_prob (','_forward_micro_batch (','_run_engine (','trajectory_credit (','compute_deltatrace','fit (','update_policy (','_optimizer_step ('])
            if active:print(line[:750])
    paths=list(Path('/tmp/ray').glob(f"session_*_{j['pid']}/logs/worker*.out"))
    for p in paths:
        with p.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-20000));lines=f.read().decode(errors='replace').splitlines()
        lines=[s for s in lines if any(x in s for x in ['[DT','[owner_trajectory]','Initial validation','[actor','batch='])]
        if lines:print(p.name,'\n'+'\n'.join(s[:1000] for s in lines[-6:]))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
