/opt/conda/bin/python - <<'PY'
from pathlib import Path
import json,time,psutil,signal,hashlib
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');a=json.loads((r/'active-training.json').read_text());j=next(j for j in a['jobs'] if j['task']=='AppWorld')
assert j['pid']==1953903 and j['observed_process_created_unix']==1791321793.72
assert j['devices']==[4,5]
assert j['source_receipt']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/appworld-fresh-native-conv-canonical-20261007-v3/appworld-dt/source.json'
assert hashlib.sha256(Path(j['source_receipt']).read_bytes()).hexdigest()=='97e3cb754f505b79a52d3dc3464b9027ae7bdc38e7074b866f80dc7624be3481'
p=psutil.Process(j['pid']);assert p.create_time()==j['observed_process_created_unix']
processes=[p]+p.children(recursive=True)
d=r/'receipts/appworld-efficiency-20261007'/('stop-before-row-prefix-'+str(int(time.time())));d.mkdir(parents=True)
q={'started_unix':time.time(),'reason':'Stop only identity-bound original v3 AppWorld tree before an independently reviewed fresh row-prefix owner submission; no checkpoint operation and no other process tree selected','current_job':j,'source_receipt_sha256':hashlib.sha256(Path(j['source_receipt']).read_bytes()).hexdigest(),'processes':[{'pid':x.pid,'created_unix':x.create_time(),'name':x.name()} for x in processes],'new_checkpoint_saves':0,'other_jobs':[{k:v for k,v in x.items() if k in ['task','pid','observed_process_created_unix','devices']} for x in a['jobs'] if x['task']!='AppWorld']}
(d/'stopping.json').write_text(json.dumps(q,indent=2)+'\n')
# Stop only the identity-bound restored job, not all Ray processes/services.
for x in [p]+list(reversed(processes[1:])):
 try:x.send_signal(signal.SIGTERM)
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(processes,timeout=8)
for x in alive:
 try:x.kill()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(alive,timeout=4)
remaining=[]
for x in alive:
 try:
  if x.status()!=psutil.STATUS_ZOMBIE:remaining.append({'pid':x.pid,'created_unix':x.create_time(),'status':x.status()})
 except psutil.NoSuchProcess:pass
q.update(observed_unix=time.time(),remaining_non_zombie=remaining)
for x in a['jobs']:
 if x['task']=='AppWorld':x.update(status='stopped_before_row_prefix_owner_fresh_submission',stop_receipt=str(d/'stopped.json'))
(r/'active-training.json').write_text(json.dumps(a,indent=2)+'\n')
(d/'stopped.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({'receipt':str(d/'stopped.json'),'remaining_non_zombie':remaining,'stopped_driver_pid':p.pid,'other_jobs':q['other_jobs'],'observed_unix':q['observed_unix']}))

PY
