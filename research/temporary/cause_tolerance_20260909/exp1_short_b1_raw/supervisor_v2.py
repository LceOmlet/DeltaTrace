"""Resume serial exp1 cells after complete byte-pinned native restoration."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,time
A=Path(__file__).resolve().parent
lock=(A/'gpu_serial.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((A/'native_restore.json').read_text())['status']=='same_previously_validated_FLA_sources_restored'
Q3='/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'
Q35=str(A/'qwen35env/bin/python')
protocol=json.loads((A/'benchmark_protocol_v2.json').read_bytes())
r={'status':'starting','runs':[],'parallel_gpu_jobs':False,
   'retained_Qwen3_core':['qwen3_dt','qwen3_ifr_multi_hop_both','qwen3_ifr_multi_hop'],
   'excluded_incomplete_environment_attempts':['qwen35_ifr_multi_hop_both','qwen35_ifr_multi_hop']}
def save():
 q=A/'queue_v2.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'queue_v2.json')
tasks=[('qwen35',m) for m in ['deltatrace_retained','ifr_multi_hop_both','ifr_multi_hop']]
for method in ['ifr_all_positions','attnlrp','perturbation_all','perturbation_CLP','perturbation_REAGENT','IG','attention_I_G']:
 for family in ['qwen3','qwen35']:tasks.append((family,method))
for family,method in tasks:
 name=family+'_'+method+'_v2';row={'family':family,'method':method,'output':name,'status':'running','started':time.time()}
 r['runs'].append(row);r['status']=name;save()
 with (A/(name+'.log')).open('w') as log:
  command=[Q3 if family=='qwen3' else Q35,'-u',str(A/'benchmark_v2.py'),'--root',str(A),'--family',family,'--method',method,'--protocol','benchmark_protocol_v2.json','--output',str(A/name)]
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
   env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'))
  row['pid']=child.pid;save()
  try:row['exit_code']=child.wait(timeout=protocol['method_timeout_seconds'])
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   row['status']='timeout'
  row['seconds']=time.time()-row['started']
 if row['status']!='timeout':
  q=A/name/'results.json';result=json.loads(q.read_text()) if q.exists() else {}
  row['status']=result.get('status','no_result');row['rows']=len(result.get('rows',[]))
  row['errors']=sum(x['status']!='ok' for x in result.get('rows',[]))
 save()
r['status']='complete';save()
