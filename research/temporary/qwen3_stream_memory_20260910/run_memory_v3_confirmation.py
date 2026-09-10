from pathlib import Path
import subprocess,json,os,time,traceback
T=Path(__file__).resolve().parent;P=json.loads((T/'memory_v3_confirmation_protocol.json').read_bytes());out=T/'memory_v3_confirmation';out.mkdir(exist_ok=False)
report={'status':'running','jobs':[],'started':time.time(),'GPU_jobs_serial':True,'independent_process_per_geometry_method_round':True}
def save():(out/'queue.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for method in P['methods']:
        for prefix in ['triton','inductor']:assert not (T/(prefix+'_memory_v3_confirmation_'+method)).exists()
    for round_index,lengths in enumerate(P['length_orders'],1):
        for index,n in enumerate(lengths):
            order=P['methods'] if (index+round_index)%2 else list(reversed(P['methods']))
            for method in order:
                name=f'r{round_index}_{method}_{n}';job={'name':name,'round':round_index,'method':method,'input_length':n,'status':'running','started':time.time()};report['jobs'].append(job);save()
                env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(T/('triton_memory_v3_confirmation_'+method)),TORCHINDUCTOR_CACHE_DIR=str(T/('inductor_memory_v3_confirmation_'+method)))
                command=['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'benchmark_memory_v3.py'),'--root','/tmp/codex_short_b1_efficiency_20260910_v1','--family','qwen3','--method',method,'--lengths',str(n),'--protocol',str(T/'memory_v3_confirmation_protocol.json'),'--output',str(out/name)]
                with (out/(name+'.log')).open('w') as log:
                    try:
                        c=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=1800);job['returncode']=c.returncode
                    except subprocess.TimeoutExpired:job['timeout_seconds']=1800
                path=out/name/'results.json';d=json.loads(path.read_bytes()) if path.exists() else {'status':'no_result_file'}
                job.update(status=d['status'],ended=time.time(),error=d.get('error'));save()
    report['status']='complete' if all(j['status']=='complete' and j.get('returncode')==0 for j in report['jobs']) else 'failed'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
finally:report['ended']=time.time();save();print(json.dumps(report),flush=True)
