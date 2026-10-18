import json,runpy,time,hashlib,statistics
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');A=R/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
owner=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
m=runpy.run_path(str(owner),run_name='historical_native_metrics_reader')
jobs=json.loads((A/'historical-version-log-identities.json').read_bytes())
out=[]
for j in jobs:
 run=Path(j['run_dir']).parent
 launch=run/'launch.json'
 spec=json.loads(launch.read_bytes()) if launch.exists() else {}
 job={**spec,**j}
 if not job.get('ray_session'):
  print(json.dumps({'identity':j['identity'],'status':'no_original_ray_session_in_snapshot','launch_keys':list(spec),'known_log':spec.get('log')}));out.append({**j,'launch':spec,'metrics':[]});continue
 records,errors=m['read_job_logs'](job);metrics,phases=m['parse_logs'](records)
 g=[x['values']['actor/grad_norm'] for x in metrics if 'actor/grad_norm' in x['values']]
 one={**j,'launch_path':str(launch),'launch_SHA256':hashlib.sha256(launch.read_bytes()).hexdigest() if launch.exists() else None,'launch':spec,'metrics':metrics,'records':records,'errors':errors,'phases':phases}
 out.append(one)
 print(json.dumps({'identity':j['identity'],'completed':metrics[-1]['step'] if metrics else None,'n':len(metrics),'grad_first5':g[:5],'grad_last5':g[-5:],'grad_max':max(g) if g else None,'grad_median':statistics.median(g) if g else None,'launch_exists':launch.exists()}))
p=A/'historical-version-native-log-metrics-v1.json'
p.write_text(json.dumps({'collected':time.time(),'owner':str(owner),'owner_SHA256':hashlib.sha256(owner.read_bytes()).hexdigest(),'jobs':out},indent=2)+'\n')
print(json.dumps({'receipt':str(p)}))
