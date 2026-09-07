"""Resolve FA wrapper work in an existing real B4 trace, without model calls."""
import collections,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent
s=json.loads((A/'fa_compact_gqa_integration_summary_20260907.json').read_text())
f=A/'snapshot${ARTIFACT_ROOT}/codex_fa_compact_gqa_integration_20260907_v1/compact_B4_profile.json'
assert hashlib.sha256(f.read_bytes()).hexdigest()==s['actual_B4_profile']['sha256']
events=json.loads(f.read_text())['traceEvents']
scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name']=='ATTR_VENDOR_FA_FINITE_P1']
assert len(scopes)==36
cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat') in ['cpu_op','user_annotation'] and 'External id' in e.get('args',{})}
groups=collections.defaultdict(lambda:{'calls':0,'GPU_seconds':0.0})
in_scope=[]
for e in events:
    if e.get('cat')!='kernel':continue
    c=cpu.get(e.get('args',{}).get('External id'))
    inside=c is not None and any(x['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=x['ts']+x['dur']+1e-3 for x in scopes)
    if 'deltatrace_fa_finite_p1_kernel' in e['name']:
        phase=e['name'].split('kernel<',1)[1].split(',',1)[0];key='finite_phase_'+phase
    elif inside:
        key='FA_wrapper:'+c['name'];in_scope.append({'name':e['name'],'cpu':c['name'],'dur':e['dur']})
    else:continue
    groups[key]['calls']+=1;groups[key]['GPU_seconds']+=e['dur']/1e6
out={'status':'existing_real_B4_trace_analyzed','model_calls':0,
     'profile_sha256':s['actual_B4_profile']['sha256'],'FA_scopes':36,'partition':dict(groups),
     'wrapper_kernel_names':dict(collections.Counter(x['name'] for x in in_scope)),
     'scope':'Summed GPU kernel durations within 36 finite FA scopes; not wall time or a general benchmark.'}
(A/'fa_gqa_remaining_profile_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='wrapper_kernel_names'},indent=2))
