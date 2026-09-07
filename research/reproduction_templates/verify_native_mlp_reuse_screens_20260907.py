"""Independently check both bounded cache screens and actual kernel dispatch."""
import hashlib,json,statistics,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
out={'status':'verified_bounded_engineering_screen','fresh_attributions':0,'fresh_quality_queries':0,
     'native_vjps':0,'runs':[],'goal_complete':False}
private_numeric=[]
for version in (1,2):
    f=A/f'snapshot${ARTIFACT_ROOT}/codex_native_mlp_reuse_screen_20260907_v{version}'
    with zipfile.ZipFile(f/'review_bundle.zip') as z:
        for name in z.namelist():
            assert '/' not in name and '\\' not in name and name not in ('.','..')
            (f/name).write_bytes(z.read(name))
    d=json.loads((f/'results.json').read_text());p=d['protocol']
    assert p==json.loads((f/'protocol.json').read_text())
    assert sha(f/'study.py')==p['study_sha256']
    for name,digest in p['sources'].items():assert sha(f/name)==digest
    assert d['status']==('screen_stopped_at_gate' if version==1 else 'bounded_screen_complete_pending_review')
    assert d['checkpoint_before']==d['checkpoint_after']
    assert d['native_sources_before']==d['native_sources_after']
    assert d['native_vjps']==d['evaluation_forwards']==d['ft_attribution_forwards']==0
    assert d['fresh_attribution_calls']==(2 if version==1 else 10)
    assert d['native_root_forwards']==d['fresh_attribution_calls']
    assert d['extra_layer_replay_calls']==d['extra_native_fa_attention_calls']==d['finite_FA_calls_enqueued']==36*d['fresh_attribution_calls']
    expected_layer_start=0 if version==1 else 4
    assert p.get('first_cached_layer',0)==expected_layer_start
    summary={'version':version,'status':d['status'],'raw_sha256':sha(f/'results.json'),
        'protocol_sha256':sha(f/'protocol.json'),'fresh_attributions':d['fresh_attribution_calls'],
        'job_seconds':d['elapsed_seconds'],'cases':[]}
    if version==2:assert p['previous_screen_raw_sha256']==out['runs'][0]['raw_sha256']
    calls=[]
    for r in d['records']:
        modes={m:[x['result'] for x in r['runs'] if x['mode']==m] for m in ['finite','sac_mlp']}
        assert len(modes['finite'])==len(modes['sac_mlp'])
        row={k:r[k] for k in ['dataset','idx','N','prompt_len','input_ids_sha256']}
        row['actual_ordinary_reference_peak']=r['ordinary_reference']['peak_allocated_bytes']
        row['memory_limit']=row['actual_ordinary_reference_peak']+72500000
        row['checks']=[]
        for old,new in zip(modes['finite'],modes['sac_mlp']):
            assert old['signed_full_sequence']==new['signed_full_sequence']
            assert old['score']==new['score'] and old['endpoint_scores32']==new['endpoint_scores32']
            for val in (old,new):
                cost=val['end_to_end_cost'];calls.append(cost)
                assert cost['native_forwards']==1 and cost['vjps']==0 and cost['native_forward_trajectories']==2
                assert cost['extra_replay_calls']==36 and cost['extra_replay_trajectories']==72
                assert cost['actual_root_input_ids']==old['end_to_end_cost']['actual_root_input_ids']
                packed=cost['actual_root_input_ids'][0];assert len(packed)==2 and len(packed[0])==len(packed[1])==r['N']
                baseline=packed[1][:]
                for j in r['eligible_positions']:baseline[j]=151645
                assert baseline==packed[0]
                assert np.array_equal(np.maximum(np.array(val['signed_full_sequence'])[r['user_positions']],0).astype(np.float32),val['score'])
                assert cost['public_FA_activity']['auxiliary_completed']==36
                assert sum(a['calls_enqueued'] for a in cost['finite_FA_activity'])==36
                assert all(c['native_input_exact'] and c['native_output_exact'] for c in val['native_layer_boundary_checks']['paired_batch'])
            reuse=new['end_to_end_cost']['native_MLP_reuse']
            assert reuse['installed_source_sha256']==p['torch_checkpoint_source_sha256']=='255ec2eccfa184eb3e13ef4791955d1f6f010d8ef1f59e7a8eccb9da26161314'
            assert reuse['saved_mm_outputs']==reuse['reused_mm_outputs']==3*(36-expected_layer_start)
            assert reuse['retained_tensor_bytes']==0
            assert [v['layer'] for v in reuse['regions']]==list(range(expected_layer_start,36))
            for rec in reuse['regions']:
                assert rec['completed'] and rec['input_exact'] and rec['save_calls']==rec['reuse_calls']==3
                assert [v['projection'] for v in rec['operations']]==['gate','up','down']
                assert rec['output_bytes']==sum(v['output_bytes'] for v in rec['operations'])
                # Actual cache is only B*N by fixed feature widths; no NxN operand.
                assert all(v['input_shape'][0]==2*r['N'] and v['input_shape'][1] in [4096,12288] for v in rec['operations'])
            row['checks'].append({'signed_and_projected_exact_to_same_job_old':True,
                'old_peak':old['peak_allocated_bytes'],'new_peak':new['peak_allocated_bytes'],
                'new_excess_above_limit':new['peak_allocated_bytes']-row['memory_limit'],
                'maximum_extra_retained_tensor_bytes':reuse['maximum_retained_tensor_bytes']})
        row['measured']={m:[v['seconds'] for v in modes[m][1:]] for m in modes}
        row['warmup']={m:modes[m][0]['seconds'] for m in modes}
        if version==2:
            assert all(v['new_excess_above_limit']<=0 for v in row['checks'])
            row['candidate_to_old_ratio']=statistics.median(row['measured']['sac_mlp'])/statistics.median(row['measured']['finite'])
            assert row['candidate_to_old_ratio']==r['candidate_to_previous_P1_ratio']
            row['speed_claim_scope']='NI2 two measured pairs; MH5 one measured pair, provisional only. No new FT timing comparison.'
        else:
            assert row['checks'][0]['new_excess_above_limit']>0
            row['speed_claim_scope']='No measured steady pair; old warm includes compilation and candidate warm includes profiler. Do not compare these as speed.'
        row['historical_quality_reuse_allowed']=all(c['new_score_matches_prior'] for c in r['checks'])
        row['historical_comparison_checks']=r['checks']
        summary['cases'].append(row)
        private_numeric.append({'version':version,**{k:r[k] for k in ['dataset','idx','runs','checks']}})
    assert len(calls)==d['fresh_attribution_calls']
    summary['sum_attribution_timer_seconds']=sum(c['seconds'] for c in calls)
    trace=f/'candidate_warm_profile.json';pr=d['records'][0]['profile'];assert sha(trace)==pr['sha256']
    es=json.loads(trace.read_text())['traceEvents']
    kernels=[e for e in es if e.get('cat')=='kernel']
    gm=[e for e in kernels if 'gemm' in e['name'].lower()]
    fa=[e for e in kernels if 'flash_fwd_kernel' in e['name']]
    finite=[e for e in kernels if 'deltatrace' in e['name'].lower() or 'finite' in e['name'].lower()]
    scopes=[e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='ATTR_NATIVE_HALF_LINEAR']
    assert len(scopes)==253 and len(fa)==len(finite)==108
    assert len(gm)==(651 if version==1 else 663)
    summary['actual_profile']={'sha256':sha(trace),'GPU_GEMM_kernels':len(gm),
        'native_FA_forward_kernels':len(fa),'finite_FA_kernels':len(finite),
        'native_half_projection_scopes':len(scopes),
        'unmodified_reference_NI0_GEMM_kernels':759,
        'note':'Same-shape NI2 v2 vs v1 restores12 GEMMs for4 uncached MLPs. Native FA remains108; this optimization does not replace attention.'}
    out['fresh_attributions']+=d['fresh_attribution_calls']
    out['runs'].append(summary)
assert out['fresh_attributions']==12
assert out['runs'][0]['cases'][0]['checks'][0]['maximum_extra_retained_tensor_bytes']-out['runs'][1]['cases'][0]['checks'][0]['maximum_extra_retained_tensor_bytes']==650641408
out['conclusion']='Official SAC reuse preserves whole signed vectors against same-job unchanged P1 and reduces actual GEMMs. Full-layer cache rejected after2calls; last32 cache passes two-case engineering screen within total12calls. Limited speed gain, sizeable linear-memory tradeoff, no new FT or quality result; do not inherit historical curves when projected-score identity fails.'
(A/'native_mlp_reuse_screens_summary_20260907.json').write_text(json.dumps(out,indent=2))
# No raw model input IDs are included in the public numerical sidecar.
for row in private_numeric:
    for run in row['runs']:
        run['result']['end_to_end_cost'].pop('actual_root_input_ids')
(A/'native_mlp_reuse_screens_numeric_20260907.json').write_text(json.dumps({'records':private_numeric},separators=(',',':')))
print(json.dumps(out,indent=2))
