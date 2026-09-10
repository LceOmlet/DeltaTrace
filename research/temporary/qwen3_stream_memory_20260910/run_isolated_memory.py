from pathlib import Path
import subprocess,json,os,time,traceback
T=Path(__file__).resolve().parent;P=json.loads((T/'isolated_memory_protocol.json').read_bytes())
out=T/'isolated_memory';out.mkdir(exist_ok=False);report={'status':'running','jobs':[],'started':time.time(),'GPU_jobs_serial':True}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for index,n in enumerate(P['input_lengths']):
  for method in (P['methods'] if index%2==0 else list(reversed(P['methods']))):
   name=f'{method}_{n}';job={'name':name,'method':method,'input_length':n,'status':'running','started':time.time()};report['jobs'].append(job);save()
   env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/('triton_memory_confirmation_'+method)),TORCHINDUCTOR_CACHE_DIR=str(T/('inductor_memory_confirmation_'+method)))
   assert Path(env['TRITON_CACHE_DIR']).exists()
   command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'benchmark_isolated_memory.py'),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method',method,'--lengths',str(n),'--protocol',str(T/'isolated_memory_protocol.json'),'--output',str(out/name)]
   with (out/(name+'.log')).open('w') as log:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=1800)
   result=json.loads((out/name/'results.json').read_bytes());job.update(status=result['status'],returncode=completed.returncode,ended=time.time());save()
   assert completed.returncode==0 and result['status']=='complete',job
 report['status']='complete'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
finally:report['ended']=time.time();save();print(json.dumps(report),flush=True)
