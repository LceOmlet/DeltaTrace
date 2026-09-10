"""Serial execution only; each method owns a fresh model process."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,time,traceback

A=Path(__file__).resolve().parent
lock=(A/'gpu_serial.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Q3='/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'
Q35=str(A/'qwen35env/bin/python')
protocol=json.loads((A/'benchmark_protocol.json').read_bytes())
r={'status':'waiting_initial_DT','runs':[],'parallel_gpu_jobs':False}
def save():
    q=A/'queue.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'queue.json')
save()
while True:
    q=A/'qwen3_dt/results.json'
    initial=json.loads(q.read_text()) if q.exists() else {}
    proc=Path('/proc/8904/status')
    ended=not proc.exists() or '\nState:\tZ' in proc.read_text()
    if initial.get('status') in ['complete','failed'] and ended:break
    time.sleep(5)
r['runs'].append({'family':'qwen3','method':'deltatrace_retained','output':'qwen3_dt','status':initial['status'],'already_launched_pid':8904})
tasks=[]
for family in ['qwen3','qwen35']:
    for method in ['ifr_multi_hop_both','ifr_multi_hop']:
        tasks.append((family,method))
    if family=='qwen35':tasks.append((family,'deltatrace_retained'))
for method in ['ifr_all_positions','attnlrp','perturbation_all','perturbation_CLP','perturbation_REAGENT','IG','attention_I_G']:
    for family in ['qwen3','qwen35']:tasks.append((family,method))
for family,method in tasks:
    name=family+'_'+method
    row={'family':family,'method':method,'output':name,'status':'running','started':time.time()}
    r['runs'].append(row);r['status']=name;save()
    with (A/(name+'.log')).open('w') as log:
        command=[Q3 if family=='qwen3' else Q35,'-u',str(A/'benchmark.py'),'--root',str(A),'--family',family,'--method',method,'--output',str(A/name)]
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'))
        row['pid']=child.pid;save()
        try:
            row['exit_code']=child.wait(timeout=protocol['method_timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            row['status']='timeout'
        row['seconds']=time.time()-row['started']
    if row['status']!='timeout':
        q=A/name/'results.json'
        result=json.loads(q.read_text()) if q.exists() else {}
        row['status']=result.get('status','no_result');row['rows']=len(result.get('rows',[]))
        row['errors']=sum(x['status']!='ok' for x in result.get('rows',[]))
    save()
r['status']='complete';save()
