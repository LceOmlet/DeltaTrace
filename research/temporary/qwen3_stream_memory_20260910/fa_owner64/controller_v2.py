from pathlib import Path
import json,os,subprocess,time
A=Path(__file__).resolve().parent;env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
p={'status':'running','stages':[]};save=lambda:(A/'controller_v2.json').write_text(json.dumps(p,indent=2)+'\n')
save()
for script,python in [('study.py','/opt/conda/bin/python'),('finite_operator.py','/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python')]:
    t=time.time()
    with (A/(script+'.log')).open('w') as log:c=subprocess.run([python,'-u',str(A/script)],stdout=log,stderr=subprocess.STDOUT,env=env)
    r=json.loads((A/('results.json' if script=='study.py' else 'operator_results.json')).read_bytes())
    p['stages'].append({'script':script,'returncode':c.returncode,'status':r['status'],'seconds':time.time()-t});save()
    if c.returncode or r['status'] not in ('finite_extension_compiled_not_executed','complete'):p['status']='failed';break
else:p['status']='complete'
save()
