from pathlib import Path
import json,os,subprocess,time
p=Path('/tmp/codex_short_b1_optimization_20260910_v1');base=Path('/tmp/codex_short_b1_efficiency_20260910_v1');r={'status':'starting','runs':[]}
def save():(p/'profile_queue_v2.json').write_text(json.dumps(r,indent=2)+'\n')
for family in ['qwen35','qwen3']:
 python=str(base/'qwen35env/bin/python') if family=='qwen35' else '/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'
 row={'family':family,'started':time.time()};r['runs'].append(row);r['status']=family;save()
 with (p/(family+'_profile_v2.log')).open('w') as log:
  child=subprocess.Popen([python,'-u',str(p/'short_b1_profile.py'),'--root',str(base),'--family',family,'--method','deltatrace_retained','--lengths','128','--protocol','benchmark_protocol_v2.json','--output',str(p/(family+'_profile_v2'))],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'))
  row['pid']=child.pid;save();row['exit_code']=child.wait(timeout=1200)
 row['seconds']=time.time()-row['started'];row['status']=json.loads((p/(family+'_profile_v2/results.json')).read_bytes())['status'];save()
r['status']='complete';save()
