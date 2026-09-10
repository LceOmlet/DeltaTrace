from pathlib import Path
import subprocess,json,os,time,traceback
T=Path(__file__).resolve().parent;P=json.loads((T/'memory_v3_rollout_protocol.json').read_bytes())
out=T/'memory_v3_rollout';out.mkdir(exist_ok=False)
report={'status':'running','jobs':[],'started':time.time(),'GPU_jobs_serial':True}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for index,n in enumerate(P['output_lengths']):
  methods=P['methods'] if index%2==0 else list(reversed(P['methods']))
  for method in methods:
   name=f'{method}_{n}';job={'name':name,'method':method,'output_length':n,'status':'running','started':time.time()}
   report['jobs'].append(job);save()
   env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/('triton_memory_v3_rollout_'+method)),TORCHINDUCTOR_CACHE_DIR=str(T/('inductor_memory_v3_rollout_'+method)))
   if index==0:assert not Path(env['TRITON_CACHE_DIR']).exists() and not Path(env['TORCHINDUCTOR_CACHE_DIR']).exists()
   command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'benchmark_memory_v3_rollout.py'),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method',method,'--lengths',str(n),'--protocol',str(T/'memory_v3_rollout_protocol.json'),'--output',str(out/name)]
   with (out/(name+'.log')).open('w') as log:
    try:
     completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=1800)
     job['returncode']=completed.returncode
    except subprocess.TimeoutExpired:job['timeout_seconds']=1800
   path=out/name/'results.json'
   result=json.loads(path.read_bytes()) if path.exists() else {'status':'no_result_file'}
   job.update(status=result['status'],ended=time.time(),row_statuses=[r['status'] for r in result.get('rows',[])],error=result.get('error'));save()
 report['status']='complete'
except BaseException:
 report['status']='failed';report['error']=traceback.format_exc()
finally:
 report['ended']=time.time();save();print(json.dumps(report),flush=True)
