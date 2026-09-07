"""Separate compiler warm work from later real per-layer FA preparation calls."""
import collections,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent
s=json.loads((A/'fa_input_preparation_integration_summary_20260907.json').read_text())
f=A/'snapshot${ARTIFACT_ROOT}/codex_fa_input_preparation_integration_20260907_v1/fused_prepare_B4_profile.json'
assert hashlib.sha256(f.read_bytes()).hexdigest()==s['actual_B4_profile']['sha256']
events=json.loads(f.read_text())['traceEvents']
scopes=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name']=='ATTR_VENDOR_FA_FINITE_P1'],key=lambda e:e['ts'])
cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat') in ['cpu_op','user_annotation'] and 'External id' in e.get('args',{})}
rows=[{'execution_index':i,'wrapper_kernel_count':0,'kernel_names':collections.Counter(),'GPU_seconds':0.0} for i in range(len(scopes))]
for e in events:
    if e.get('cat')!='kernel' or 'deltatrace_fa_finite_p1_kernel' in e['name']:continue
    c=cpu.get(e.get('args',{}).get('External id'))
    if c is None:continue
    matching=[i for i,x in enumerate(scopes) if x['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=x['ts']+x['dur']+1e-3]
    assert len(matching)<2
    if not matching:continue
    r=rows[matching[0]];r['wrapper_kernel_count']+=1;r['kernel_names'][e['name']]+=1;r['GPU_seconds']+=e['dur']/1e6
assert len(rows)==36 and sum(r['wrapper_kernel_count'] for r in rows)==s['FA_wrapper_profile']['GPU_kernel_count']
out={'status':'existing_profile_warm_and_later_scopes_separated','profile_sha256':s['actual_B4_profile']['sha256'],
     'first_scope':rows[0],'later_scope_kernel_counts':dict(collections.Counter(r['wrapper_kernel_count'] for r in rows[1:])),
     'later_scopes':rows[1:],'model_calls':0,
     'scope':'All profiled warm execution retained. Later scopes are actual later layers, not a new steady latency benchmark; first scope includes compilation/autotuning.'}
(A/'fa_input_preparation_scope_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='later_scopes'},indent=2))
