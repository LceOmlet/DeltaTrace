"""Preserve actual formal step 2 and its original optimizer observation."""
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, REPO, SCP, SSH

remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,time,re,hashlib
root=Path('@ROOT@');job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='TextCraft')
assert job['pid']==212110
out=root/'receipts/owner-b8-dispatch-20260930'
optimizer_path=out/'textcraft-optimizer-observed-1790788320.json'
optimizer=json.loads(optimizer_path.read_text());assert optimizer['driver_pid']==job['pid']
records=[];dt=[];progress=[]
for path in Path('/tmp/ray').glob(f"session_*_{job['pid']}/logs/worker*.*"):
 if path.suffix not in ('.out','.err'):continue
 reports=[];rounds=[]
 with path.open(errors='replace') as stream:
  for line in stream:
   line=re.sub(r'\x1b\[[0-9;]*[mA]','',line).strip()
   if line.startswith('step:2 - '):records.append(dict(path=str(path),line=line))
   if line.startswith('[DeltaTrace readout] '):reports.append(json.loads(line.split('] ',1)[1]))
   if line.startswith('Rounds '):rounds.append(line)
 if len(reports)>=2:
  dt.append(dict(path=str(path),report={k:v for k,v in reports[1].items() if k not in ('traces','minimum_log_ratio_batch','actual_row_lengths')}))
 if rounds:progress.append(dict(path=str(path),latest=rounds[-4:]))
assert len(records)==1 and len(dt)==2,(len(records),len(dt))
metrics={key:float(value) for key,value in re.findall(r'(?:^| - )([\w/]+):([^ ]+)',records[0]['line'])}
source_receipt=Path(job.get('source_receipt',str(Path(job['output'])/'source.json')))
record=dict(task='TextCraft',formal_step=2,observed_unix=time.time(),driver_pid=job['pid'],
 source_receipt=str(source_receipt),source_sha256=hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
 entry=job['entry'],verl_root=job['verl_root'],dt_root=job['dt_root'],
 metrics=metrics,metric_precision='Original console formatting; rounded values are not exact internal scalars.',
 raw_metrics=records[0],dt=dt,optimizer=optimizer,progress=progress,
 scope='Two actual formal iterations and native optimizer state; not a whole-network numerical tolerance test or proof of reward improvement.',
 checkpoint_markers=[dict(path=str(p),value=p.read_text()) for p in Path(job['checkpoints']).rglob('latest_checkpointed_iteration.txt')])
(out/'textcraft-second-formal-update.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(metrics=metrics,progress=progress,optimizer_steps=[w['optimizer_steps'] for rank in optimizer['workers'] for w in rank]),indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
local=AUDIT/'formal-progress-20261001';local.mkdir(exist_ok=True)
source=f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/textcraft-second-formal-update.json'
subprocess.run(SCP+[source,str(local/'textcraft-second-formal-update.json')],check=True)
subprocess.run(SCP+[source,str(REPO/'experiments/rl/results_textcraft_continuity.json')],check=True)
