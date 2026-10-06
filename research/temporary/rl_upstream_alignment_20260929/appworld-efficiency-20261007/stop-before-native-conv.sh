/opt/conda/bin/python - <<'PY'
from pathlib import Path
import json,time,psutil,signal,hashlib
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');a=json.loads((r/'active-training.json').read_text());j=next(j for j in a['jobs'] if j['task']=='AppWorld')
assert j['pid']==722849 and j['observed_process_created_unix']==1791310127.33
p=psutil.Process(j['pid']);assert p.create_time()==j['observed_process_created_unix']
processes=[p]+p.children(recursive=True)
d=r/'receipts/appworld-efficiency-20261007'/('stop-before-native-conv-'+str(int(time.time())));d.mkdir(parents=True)
q={'started_unix':time.time(),'reason':'Accepted minimal cached-convolution API patch passed original primitive tests and B8 long/32768 capacity; stop only this base-initialized AppWorld tree before deploying measured owner changes; no checkpoint load or save','current_job':j,'source_receipt_sha256':hashlib.sha256(Path(j['source_receipt']).read_bytes()).hexdigest(),'processes':[{'pid':x.pid,'created_unix':x.create_time(),'name':x.name()} for x in processes],'new_checkpoint_saves':0,'other_jobs':[{k:v for k,v in x.items() if k in ['task','pid','observed_process_created_unix','devices']} for x in a['jobs'] if x['task']!='AppWorld']}
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
 if x['task']=='AppWorld':x.update(status='stopped_before_native_conv_owner_deployment',stop_receipt=str(d/'stopped.json'))
(r/'active-training.json').write_text(json.dumps(a,indent=2)+'\n')
(d/'stopped.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({'receipt':str(d/'stopped.json'),'remaining_non_zombie':remaining,'stopped_driver_pid':p.pid,'other_jobs':q['other_jobs'],'observed_unix':q['observed_unix']}))

PY
