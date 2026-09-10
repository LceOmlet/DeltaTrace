from pathlib import Path
import subprocess,json,os,time,traceback
T=Path(__file__).resolve().parent;P=json.loads((T/'memory_confirmation_protocol.json').read_bytes())
out=T/'memory_confirmation';out.mkdir(exist_ok=False)
report={'status':'running','jobs':[],'started':time.time()}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for ri,methods in enumerate(P['method_rounds']):
  for method in methods:
   name=f'round{ri+1}_{method}';job={'name':name,'method':method,'round':ri+1,'status':'running','started':time.time()}
   report['jobs'].append(job);save()
   env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/('triton_memory_confirmation_'+method)),TORCHINDUCTOR_CACHE_DIR=str(T/('inductor_memory_confirmation_'+method)))
   if ri==0:assert not Path(env['TRITON_CACHE_DIR']).exists() and not Path(env['TORCHINDUCTOR_CACHE_DIR']).exists()
   command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'benchmark_memory_confirmation.py'),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method',method,'--lengths',','.join(map(str,P['length_orders'][ri])),'--protocol',str(T/'memory_confirmation_protocol.json'),'--output',str(out/name)]
   with (out/(name+'.log')).open('w') as log:
    completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=3600)
   result=json.loads((out/name/'results.json').read_bytes())
   job.update(status=result['status'],returncode=completed.returncode,ended=time.time());save()
   assert completed.returncode==0 and result['status']=='complete',job
 report['status']='complete'
except BaseException:
 report['status']='failed';report['error']=traceback.format_exc()
finally:
 report['ended']=time.time();save();print(json.dumps(report),flush=True)
