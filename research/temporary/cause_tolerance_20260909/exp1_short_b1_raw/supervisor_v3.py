"""Rerun the three original perturbation methods after restoring their real model."""
from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time
A=Path(__file__).resolve().parent
revision='301e6a42cb0d9976a6d6a26a079fef81c18aa895'
cache=Path('/root/.cache/huggingface/hub/models--allenai--longformer-base-4096')
protocol=json.loads((A/'benchmark_protocol_v3.json').read_bytes())
r={'status':'waiting_for_v2_and_verified_aux_cache','runs':[],'parallel_gpu_jobs':False}
def save():
 p=A/'queue_v3.partial';p.write_text(json.dumps(r,indent=2));p.replace(A/'queue_v3.json')
save()
while True:
 q=json.loads((A/'queue_v2.json').read_bytes())
 ref=cache/'refs/main'
 if q['status']=='complete' and ref.exists() and ref.read_text().strip()==revision:break
 time.sleep(5)
lock=(A/'gpu_serial.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
verified={}
for record in protocol['auxiliary_model']['files']:
 path=cache/'snapshots'/revision/record['name'];assert path.stat().st_size==record['bytes']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==record['sha256'];verified[record['name']]=h.hexdigest()
r['verified_auxiliary_files']=verified;save()
for method in ['perturbation_all','perturbation_CLP','perturbation_REAGENT']:
 for family in ['qwen3','qwen35']:
  name=family+'_'+method+'_v3'
  row={'family':family,'method':method,'output':name,'status':'running','started':time.time()}
  r['runs'].append(row);r['status']=name;save()
  python='/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python' if family=='qwen3' else str(A/'qwen35env/bin/python')
  command=[python,'-u',str(A/'benchmark_v2.py'),'--root',str(A),'--family',family,'--method',method,'--protocol','benchmark_protocol_v3.json','--output',str(A/name)]
  with (A/(name+'.log')).open('w') as log:
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
   result=json.loads((A/name/'results.json').read_bytes())
   row.update(status=result['status'],rows=len(result['rows']),errors=sum(x['status']!='ok' for x in result['rows']))
  save()
r['status']='complete';save()
