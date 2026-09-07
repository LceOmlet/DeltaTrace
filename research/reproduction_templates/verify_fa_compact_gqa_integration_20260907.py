import hashlib,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_compact_gqa_integration_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
with zipfile.ZipFile(F/'review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (F/name).write_bytes(z.read(name))
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert p==json.loads((F/'protocol.json').read_text())
assert sha(F/'study.py')==p['study_sha256']
for name,digest in p['sources'].items():assert sha(F/name)==digest
assert d['status']=='compact_GQA_B1_and_real_B4_exact'
assert d['checkpoint_before']==d['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']
assert d['fresh_attributions']==d['native_root_forwards']==8
assert d['native_vjps']==d['evaluation_forwards']==d['ft_attribution_forwards']==d['quality_queries']==0
assert d['extra_layer_replay_calls']==d['extra_native_fa_attention_calls']==d['finite_FA_calls_enqueued']==288
assert d['native_attribution_endpoint_trajectories']==40
out={'status':'compact_GQA_B1_and_real_B4_verified','raw_sha256':sha(F/'results.json'),
    'protocol_sha256':sha(F/'protocol.json'),'attributions':8,'quality_queries':0,'VJPs':0,
    'native_root_forwards':8,'native_endpoint_trajectories':40,'decoder_replays':288,
    'auxiliary_public_FA_calls':288,'finite_attention_calls':288,
    'groups':[],'job_seconds':d['job_seconds'],'goal_complete':False}
numeric=[]
for index,g in enumerate(d['integration_groups']):
    assert g['selection']==p['integration_groups'][index]
    batch=g['example_batch_size'];assert batch in [1,4]
    assert len(g['runs'])==4
    for repeat in range(2):
        old,new=[next(r['result'] for r in g['runs'] if r['mode']==mode and r['repeat']==repeat) for mode in ['old','compact']]
        assert np.array_equal(old['signed_full_sequence'],new['signed_full_sequence'])
        assert old['endpoint_scores32']==new['endpoint_scores32']
        for mode,result in [('old',old),('compact',new)]:
            cost=result['end_to_end_cost']
            assert cost['native_forwards']==1 and cost['native_forward_trajectories']==batch*2
            assert cost['extra_replay_calls']==36
            assert cost['public_FA_activity']['auxiliary_attempts']==cost['public_FA_activity']['auxiliary_completed']==36
            for item in cost['public_FA_activity']['metadata']:
                assert item['testing_return']['numel'] in [0,None]
            boundaries=result['native_layer_boundary_checks']
            if isinstance(boundaries,dict):boundaries=boundaries['paired_batch']
            for item in boundaries:
                assert item['native_input_exact'] and item['native_output_exact']
            finite=cost['finite_FA_activity'];assert len(finite)==36
            for item in finite:
                assert item['calls_attempted']==item['calls_enqueued']==1
                shapes=[v['shape'] for v in item['buffer_contract']]
                assert shapes[0][0]==batch and shapes[0][1]==32
                for pos in [1,3,4,7]:assert shapes[pos][1]==(8 if mode=='compact' else 32)
                assert shapes[12]==shapes[13]==shapes[14]==shapes[0]
                if mode=='compact':assert item['GQA_input_expansion'] is False and item['kv_heads']==8
        assert g['comparisons'][repeat]['full_signed_exact'] and g['comparisons'][repeat]['endpoints_exact']
    old,new=[next(r['result'] for r in g['runs'] if r['mode']==mode and not r['warm']) for mode in ['old','compact']]
    out['groups'].append({'selection':g['selection'],'example_batch_size':batch,
        'all_same_job_full_vectors_exact':True,'endpoints_exact':True,'sign_flips':0,
        'old_measured_seconds':old['seconds'],'compact_measured_seconds':new['seconds'],
        'compact_to_old_ratio':new['seconds']/old['seconds'],
        'old_peak_bytes':old['peak_allocated_bytes'],'compact_peak_bytes':new['peak_allocated_bytes'],
        'timing_scope':'One measured pair after warmup; no stable performance or FT superiority claim. NI2 old-path1.84s is unusually high against prior same-shape~0.99s and is not attributed causally to GQA savings.'})
    numeric.append(g)
profile=d['compact_B4_profile'];assert sha(F/profile['file'])==profile['sha256']
events=json.loads((F/profile['file']).read_text())['traceEvents']
kernels=Counter(e['name'] for e in events if e.get('cat')=='kernel')
finite={k:n for k,n in kernels.items() if 'deltatrace_fa_finite_p1_kernel' in k}
native={k:n for k,n in kernels.items() if 'flash' in k.lower() and 'fwd' in k.lower() and k not in finite}
assert sum(finite.values())==108,finite
assert sum(native.values())==108,native
out['actual_B4_profile']={'sha256':profile['sha256'],'finite_kernels':finite,'native_FA_kernels':native,
    'no_new_dense_attention_source_path':True}
out['conclusion']='Compact GQA input integration passes actual B1 and B4 with unchanged full signed vectors and native endpoints. Default FA plus finite FA remains actual dispatched. Input copies removed; endpoint-average/layout, grouped output reduction and three passes remain. No new curves or full FT speed comparison.'
(A/'fa_compact_gqa_integration_summary_20260907.json').write_text(json.dumps(out,indent=2))
(A/'fa_compact_gqa_integration_numeric_20260907.json').write_text(json.dumps({'groups':numeric},separators=(',',':')))
print(json.dumps(out))
