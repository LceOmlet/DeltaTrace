"""Collect original 85-iteration logs and passive readout summaries; no model calls."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RUN = ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
LOGROOT = Path('/tmp/ray/session_2026-10-12_16-27-07_350166_3172194/logs')
plot_path = OUT/'original_plot_training_progress.py'
plot = runpy.run_path(str(plot_path),run_name='original_plot_read_only')
records, errors = plot['read_job_logs']({'ray_session':str(LOGROOT.parent)})
metrics, phases = plot['parse_logs'](records)
reports = {}
for path in LOGROOT.glob('worker-*.out'):
    rows = []
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        if '[DeltaTrace readout] ' in line:
            body = line.split('[DeltaTrace readout] ',1)[1]
            try: value=json.loads(body)
            except json.JSONDecodeError: continue
            if 'raw_advantage_groups' in value:
                rows.append({'source':{'path':str(path),'line':number},'value':value})
    if rows: reports[str(path)]=rows
cp=RUN/'textcraft-dt/checkpoints'
cp_files={p.name:[{'name':f.name,'bytes':f.stat().st_size} for f in (p/'actor').glob('*.pt')] for p in sorted(cp.glob('global_step_*'))}
first_rollout=RUN/'textcraft-dt/rollouts/1.jsonl'
first=json.loads(first_rollout.open().readline())
report={'unix':time.time(),'scope':__doc__,'plot_parser_SHA256':hashlib.sha256(plot_path.read_bytes()).hexdigest(),
        'metrics':metrics,'errors':errors,'raw_reports':reports,'checkpoints':cp_files,
        'rollout_row_keys':list(first),'rollout_row_sample':{k:str(v)[:150] for k,v in first.items()}}
path=OUT/'temporal-native-reports-v1.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
