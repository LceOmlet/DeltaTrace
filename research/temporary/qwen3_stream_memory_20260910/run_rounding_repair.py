from pathlib import Path
import subprocess,os,time,json,traceback
T=Path(__file__).resolve().parent;out=T/'rounding_repair';out.mkdir(exist_ok=False)
report={'status':'running','jobs':[],'started':time.time()}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for name,driver,protocol,lengths in [('rollout','benchmark_rounding_rollout.py','rounding_rollout_protocol.json','10,100,500'),('short','benchmark_rounding_short.py','rounding_short_protocol.json','128,256,512,1024')]:
  job={'name':name,'status':'running','started':time.time()};report['jobs'].append(job);save()
  env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/'triton_rollout_deltatrace_streamed'),TORCHINDUCTOR_CACHE_DIR=str(T/'inductor_rollout_deltatrace_streamed'))
  command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/driver),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method','deltatrace_streamed','--lengths',lengths,'--protocol',str(T/protocol),'--output',str(out/name)]
  with (out/(name+'.log')).open('w') as log:c=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=1800)
  d=json.loads((out/name/'results.json').read_bytes());job.update(status=d['status'],returncode=c.returncode,ended=time.time());save()
  assert c.returncode==0 and d['status']=='complete',job
 report['status']='complete'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
finally:report['ended']=time.time();save();print(json.dumps(report),flush=True)
