from pathlib import Path
import json,os,subprocess,time,traceback
A=Path(__file__).resolve().parent;plan=json.loads((A/'fa_cached_followup_plan.json').read_bytes());out=A/'fa_cached_followup';out.mkdir(exist_ok=False)
env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(A/'triton_memory_v2_confirmation_deltatrace_streamed'),TORCHINDUCTOR_CACHE_DIR=str(A/'inductor_memory_v2_confirmation_deltatrace_streamed'))
python='/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python';q={'status':'running','jobs':[],'GPU_serial':True};save=lambda:(out/'queue.json').write_text(json.dumps(q,indent=2)+'\n');save()
jobs=[('probe',None)]+[(kind,n) for kind,ns in [('short',[128,256,512]),('rollout',[10,100,500])] for n in ns]
try:
    for kind,n in jobs:
        name=kind if n is None else kind+'_'+str(n);record={'name':name,'kind':kind,'length':n,'status':'running','started':time.time()};q['jobs'].append(record);save()
        if kind=='probe':command=[python,'-u',str(A/'run_template_validation.py')];result=A/'template_validation/results.json'
        else:
            command=[python,'-u',str(A/('benchmark_fa_cached_'+kind+'.py')),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method','deltatrace_streamed','--lengths',str(n),'--protocol',str(A/('fa_cached_'+kind+'_protocol.json')),'--output',str(out/name)];result=out/name/'results.json'
        with (out/(name+'.log')).open('w') as log:c=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=2400)
        d=json.loads(result.read_bytes());record.update(returncode=c.returncode,status=d['status'],ended=time.time());save();assert c.returncode==0 and d['status']=='complete',name
    q['status']='complete'
except Exception:q['status']='failed';q['error']=traceback.format_exc()
finally:save()
