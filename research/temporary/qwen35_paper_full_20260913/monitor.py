"""Compact read-only status for the in-progress Qwen3.5 paper comparison."""
import collections,json,pathlib,time
ROOT=pathlib.Path('/mnt/geogpt-doc-new/deepresearch/lzq/deltatrace_qwen35_20260912')
OWN=pathlib.Path('/tmp/codex_source_v2_gpu_20260910_v1/qwen35_paper_full_20260913')
def read(p):
    return json.loads(p.read_bytes()) if p.exists() else {}
r=read(ROOT/'paper_recovery_dynamic/results.json')
s=read(ROOT/'paper_comparison_status.json')
c=read(OWN/'cost_v1/results.json')
d=dict(utc=time.strftime('%H:%M:%S',time.gmtime()),stage=s.get('status'),recovery=r.get('status'),
    count=len(r.get('cases',[])),tasks=dict(collections.Counter(x['dataset'] for x in r.get('cases',[]))),
    cost=c.get('status'),cost_calls=sum(len(x['calls']) for x in c.get('examples',[])))
for name,x in [('recovery',r),('controller',s),('cost',c),('cost_controller',read(OWN/'cost_controller_status.json'))]:
    if x.get('error'):d[name+'_error']=x['error'][-2000:]
print(json.dumps(d,ensure_ascii=False))
