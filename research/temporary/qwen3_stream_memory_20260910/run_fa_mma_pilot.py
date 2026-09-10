from pathlib import Path
import json,os,subprocess,time
T=Path(__file__).resolve().parent;out=T/'fa_mma_pilot';out.mkdir(exist_ok=False);r={'status':'running','jobs':[]}
def save():(out/'queue.json').write_text(json.dumps(r,indent=2)+'\n')
save()
env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/'triton_memory_v3_confirmation_deltatrace_streamed'),TORCHINDUCTOR_CACHE_DIR=str(T/'inductor_memory_v3_confirmation_deltatrace_streamed'))
for kind,n in [('short',1024),('rollout',100)]:
 j={'name':kind,'length':n,'status':'running','started':time.time()};r['jobs'].append(j);save()
 with (out/(kind+'.log')).open('w') as log:
  c=subprocess.run(['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/('benchmark_fa_mma_pilot_'+kind+'.py')),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method','deltatrace_streamed','--lengths',str(n),'--protocol',str(T/('fa_mma_pilot_'+kind+'_protocol.json')),'--output',str(out/kind)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=1800)
 d=json.loads((out/kind/'results.json').read_bytes());j.update(status=d['status'],returncode=c.returncode,ended=time.time(),error=d.get('error'));save()
 if d['status']!='complete' or c.returncode:r['status']='failed';break
else:r['status']='complete'
save();print(json.dumps(r),flush=True)
