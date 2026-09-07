"""Derive the focused verifier from the already checked FA integration verifier."""
from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'verify_fa_compact_gqa_integration_20260907.py').read_text()
s=s.replace('import hashlib,json,zipfile','import hashlib,json,zipfile,statistics')
s=s.replace('codex_fa_compact_gqa_integration_20260907_v1','codex_fa_input_preparation_integration_20260907_v1')
s=s.replace('compact_GQA_B1_and_real_B4_exact','FA_input_preparation_B1_and_real_B4_exact')
s=s.replace("d['native_root_forwards']==8", "d['native_root_forwards']==12")
s=s.replace("d['finite_FA_calls_enqueued']==288", "d['finite_FA_calls_enqueued']==432")
s=s.replace("d['native_attribution_endpoint_trajectories']==40", "d['native_attribution_endpoint_trajectories']==60")
s=s.replace("'attributions':8", "'attributions':12").replace("'native_root_forwards':8", "'native_root_forwards':12")
s=s.replace("'native_endpoint_trajectories':40", "'native_endpoint_trajectories':60")
s=s.replace("'decoder_replays':288", "'decoder_replays':432").replace("'auxiliary_public_FA_calls':288", "'auxiliary_public_FA_calls':432").replace("'finite_attention_calls':288", "'finite_attention_calls':432")
s=s.replace("len(g['runs'])==4", "len(g['runs'])==6").replace('for repeat in range(2):','for repeat in range(3):')
s=s.replace("'compact'", "'fused_prepare'")
s=s.replace("(8 if mode=='fused_prepare' else 32)", '8')
s=s.replace("if mode=='fused_prepare':assert item['GQA_input_expansion'] is False and item['kv_heads']==8", "assert item['GQA_input_expansion'] is False and item['kv_heads']==8\n                if mode=='fused_prepare':assert item['input_preparation']=='public_torch_compile_inductor_same_expressions'")
start=s.index("    old,new=[next(r['result'] for r in g['runs'] if r['mode']==mode and not r['warm'])")
end=s.index('\n    numeric.append(g)',start)
s=s[:start]+'''    measured={mode:[r['result']['seconds'] for r in g['runs'] if r['mode']==mode and not r['warm']] for mode in ['old','fused_prepare']}
    peaks={mode:[r['result']['peak_allocated_bytes'] for r in g['runs'] if r['mode']==mode and not r['warm']] for mode in ['old','fused_prepare']}
    assert all(len(v)==2 for v in measured.values())
    out['groups'].append({'selection':g['selection'],'example_batch_size':batch,
        'all_same_job_full_vectors_exact':True,'endpoints_exact':True,'sign_flips':0,
        'measured_seconds':measured,'peak_bytes':peaks,
        'fused_to_old_median_ratio':statistics.median(measured['fused_prepare'])/statistics.median(measured['old']),
        'timing_scope':'Two measured pairs per shape, same-job current compact-GQA baseline; no FT comparison.'})''' +s[end:]
s=s.replace("d['compact_B4_profile']", "d['fused_prepare_B4_profile']")
s=s.replace("'compact_GQA_B1_and_real_B4_verified'", "'FA_input_preparation_B1_and_real_B4_verified'")
s=s.replace("out['conclusion']='Compact GQA input integration passes actual B1 and B4 with unchanged full signed vectors and native endpoints. Default FA plus finite FA remains actual dispatched. Input copies removed; endpoint-average/layout, grouped output reduction and three passes remain. No new curves or full FT speed comparison.'", """scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name']=='ATTR_VENDOR_FA_FINITE_P1']
assert len(scopes)==36
cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat') in ['cpu_op','user_annotation'] and 'External id' in e.get('args',{})}
wrapper=[]
for e in events:
    if e.get('cat')!='kernel' or 'deltatrace_fa_finite_p1_kernel' in e['name']:continue
    c=cpu.get(e.get('args',{}).get('External id'))
    if c is not None and any(x['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=x['ts']+x['dur']+1e-3 for x in scopes):wrapper.append(e)
out['FA_wrapper_profile']={'GPU_kernel_count':len(wrapper),'GPU_seconds':sum(x['dur'] for x in wrapper)/1e6,
    'kernel_names':dict(Counter(x['name'] for x in wrapper)),'scope':'Existing-budget profiled B4 warm; compiler warm work may be included. Kernel time is not end-to-end latency.'}
out['conclusion']='Full signed vectors and endpoints remain exact; actual native FA and finite kernels unchanged. Report all measured latency; fewer wrapper kernels do not establish useful end-to-end speed. No new curves/FT comparison.'""")
s=s.replace('fa_compact_gqa_integration_summary_20260907.json','fa_input_preparation_integration_summary_20260907.json')
s=s.replace('fa_compact_gqa_integration_numeric_20260907.json','fa_input_preparation_integration_numeric_20260907.json')
(A/'verify_fa_input_preparation_integration_20260907.py').write_text(s)
print(A/'verify_fa_input_preparation_integration_20260907.py')
