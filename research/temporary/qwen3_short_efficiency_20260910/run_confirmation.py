from pathlib import Path
import subprocess,json,os,time
T=Path(__file__).resolve().parent;P=json.loads((T/'confirmation_protocol.json').read_bytes())
out=T/'confirmation';out.mkdir(exist_ok=False)
env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/'triton_confirmation'),TORCHINDUCTOR_CACHE_DIR=str(T/'inductor_confirmation'))
assert not Path(env['TRITON_CACHE_DIR']).exists() and not Path(env['TORCHINDUCTOR_CACHE_DIR']).exists()
report={'status':'running','jobs':[],'started':time.time()}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for round_index,methods in enumerate(P['method_rounds']):
  for method in methods:
   name=f'round{round_index+1}_{method}';job={'name':name,'method':method,'round':round_index+1,'status':'running','started':time.time()}
   report['jobs'].append(job);save()
   with (out/(name+'.log')).open('w') as log:
    command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'benchmark_confirmation.py'),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method',method,'--lengths',','.join(map(str,P['length_orders'][round_index])),'--protocol',str(T/'confirmation_protocol.json'),'--output',str(out/name)]
    completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=3600)
   result=json.loads((out/name/'results.json').read_bytes())
   job.update(status=result['status'],returncode=completed.returncode,ended=time.time());save()
   assert completed.returncode==0 and result['status']=='complete',job
 report['status']='complete'
except BaseException:
 import traceback
 report['status']='failed';report['error']=traceback.format_exc()
finally:
 report['ended']=time.time();save();print(json.dumps(report),flush=True)
