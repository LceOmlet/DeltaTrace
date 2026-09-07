"""Freeze one MLP-cache candidate and a maximum of twelve real attributions."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
A=Path(__file__).resolve().parent
bounded_adjustment='--bounded-adjustment' in sys.argv
suffix='_32' if bounded_adjustment else ''
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
original=(A/'original_FT_seq_cost16_20260907.py').read_text()
source=original[:original.index('\ndef original_seq_only(')]
source=source.replace("import contextlib\n", "import contextlib\nfrom native_mlp_selective_checkpoint_reuse import NativeMLPSelectiveReuse\n", 1)
source=source.replace("assert capture_mode in ['dense','finite']", "assert capture_mode in ['dense','finite','sac_mlp']")
source=source.replace("capture_mode=='finite'", "capture_mode in ['finite','sac_mlp']")
source=source.replace("torch.cuda.empty_cache();preprop_peak=0;finished=False", "torch.cuda.empty_cache();preprop_peak=0;finished=False;reuse_activity={}")
start=source.index('            reference,clean=capture_checkpoint_pair(')
end=source.index('\n        finished=True', start)
block=source[start:end]
source=source[:start]+"            reuse_context=NativeMLPSelectiveReuse(model,reuse_activity,first_cached_layer=p['first_cached_layer']) if capture_mode=='sac_mlp' else contextlib.nullcontext()\n            with reuse_context:\n"+'\n'.join('    '+line for line in block.splitlines())+source[end:]
source=source.replace("        full['public_FA_activity']=activity", "        full['native_MLP_reuse']=reuse_activity\n        full['public_FA_activity']=activity")
source=source.replace("if mode=='finite':", "if mode in ['finite','sac_mlp']:")
source=source.replace("result=strong_run(ex,expected,capture_mode='finite')", "result=strong_run(ex,expected,capture_mode=mode)")
source+=r'''
import statistics
import torch.utils.checkpoint as torch_checkpoint
assert hashlib.sha256(Path(torch_checkpoint.__file__).read_bytes()).hexdigest()==p['torch_checkpoint_source_sha256']
report['torch_checkpoint_source_sha256']=p['torch_checkpoint_source_sha256']
save()
try:
    stopped=False
    for case_number,(dataset,index) in enumerate(p['selection']):
        data=ROOT/f'exp/exp2/data/{dataset}.jsonl'
        assert hashlib.sha256(data.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(data)[index]
        expected=next(x for x in parent['records'] if (x['dataset'],x['idx'])==(dataset,index))
        prior=next(x for x in quality_parent['records'] if (x['dataset'],x['idx'])==(dataset,index))
        row={k:expected[k] for k in ['dataset','idx','input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions']}
        row.update(N=len(expected['input_ids']),runs=[],checks=[],ordinary_reference=prior['ordinary_backward'],profile=None)
        report['records'].append(row)
        for repeat in range(p['repeat_counts'][case_number]):
            # First warm pair compiles the unchanged finite arithmetic before the
            # candidate's passive profile. Measured pairs reverse their order.
            order=['finite','sac_mlp'] if repeat in (0,2) else ['sac_mlp','finite']
            for mode in order:
                assert report['fresh_attribution_calls']<p['maximum_fresh_attributions']
                report['active']=[dataset,index,repeat,mode];save()
                if case_number==0 and repeat==0 and mode=='sac_mlp':
                    with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
                        result=run_mode(ex,expected,mode)
                    trace=HERE/'candidate_warm_profile.json';prof.export_chrome_trace(str(trace))
                    row['profile']={'file':trace.name,'sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                        'scope':'This existing-budget candidate warm call only; excluded from measured latency.'}
                else:
                    result=run_mode(ex,expected,mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':result})
                if mode=='sac_mlp':
                    reuse=result['end_to_end_cost']['native_MLP_reuse']
                    assert reuse['installed_source_sha256']==p['torch_checkpoint_source_sha256']
                    assert reuse['saved_mm_outputs']==reuse['reused_mm_outputs']==3*(36-p['first_cached_layer'])
                    assert reuse['retained_tensor_bytes']==0
                    assert all(v['completed'] and v['input_exact'] for v in reuse['regions'])
                save();print('MLP_REUSE_SCREEN',dataset,index,repeat,mode,result['seconds'],result['peak_allocated_bytes'],flush=True)
            old=next(x['result'] for x in row['runs'] if x['repeat']==repeat and x['mode']=='finite')
            new=next(x['result'] for x in row['runs'] if x['repeat']==repeat and x['mode']=='sac_mlp')
            a=torch.tensor(old['signed_full_sequence'],dtype=torch.float64)
            b=torch.tensor(new['signed_full_sequence'],dtype=torch.float64)
            eligible=torch.tensor(expected['eligible_positions'])
            flip=torch.sign(a[eligible])!=torch.sign(b[eligible])
            rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30))
            mass=float(a[eligible][flip].abs().sum()/a[eligible].abs().sum().clamp_min(1e-30))
            memory_limit=prior['ordinary_backward']['peak_allocated_bytes']+p['memory_allowance_bytes']
            check={'repeat':repeat,'full_signed_exact':bool(torch.equal(a,b)),
                'relative_l2':rel,'sign_flip_count':int(flip.sum()),'flipped_old_absolute_mass_fraction':mass,
                'projected_score_exact':old['score']==new['score'],
                'old_score_matches_prior':old['score']==prior['scores']['finite'],
                'new_score_matches_prior':new['score']==prior['scores']['finite'],
                'memory_limit_from_verified_prior_native_backward':memory_limit,
                'new_peak_below_memory_limit':new['peak_allocated_bytes']<=memory_limit,
                'endpoint_scores_exact':old['endpoint_scores32']==new['endpoint_scores32']}
            row['checks'].append(check);save()
            if not(check['endpoint_scores_exact'] and rel<=p['relative_l2_limit'] and mass<=p['flipped_mass_limit'] and check['new_peak_below_memory_limit']):
                row['early_stop_reason']='Numerical or accepted memory boundary not passed; no wider runs.'
                stopped=True;break
        if not stopped:
            med={mode:statistics.median(x['result']['seconds'] for x in row['runs'] if x['mode']==mode and not x['warmup']) for mode in ['finite','sac_mlp']}
            row['measured_medians']=med;row['candidate_to_previous_P1_ratio']=med['sac_mlp']/med['finite']
            if row['candidate_to_previous_P1_ratio']>p['maximum_ratio_to_continue']:
                row['early_stop_reason']='No latency gain at first screen; no wider runs.';stopped=True
        row['complete']=not stopped;save()
        if stopped:break
    native_method_audit(True)
    report['native_sources_after']=native_sources()
    assert report['native_sources_before']==report['native_sources_after']
    report['checkpoint_after']=checkpoint_receipt()
    assert report['checkpoint_before']==report['checkpoint_after']
    assert report['fresh_attribution_calls']<=p['maximum_fresh_attributions']
    assert report['evaluation_forwards']==report['native_vjps']==0
    report['status']='screen_stopped_at_gate' if stopped else 'bounded_screen_complete_pending_review'
    report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]+[f.name for f in HERE.glob('*_profile.json')]:z.write(HERE/name,name)
'''
ast.parse(source)
study=A/f'native_mlp_reuse_screen{suffix}_20260907.py';study.write_text(source,encoding='utf-8')
p=json.loads((A/'original_FT_seq_cost16_protocol_20260907.json').read_text())
p['sources']['native_mlp_selective_checkpoint_reuse.py']=sha(A/'native_mlp_selective_checkpoint_reuse.py')
p.update(study_sha256=sha(study),selection=[['niah_mq_q2',2],['morehopqa',5]],
    first_cached_layer=4 if bounded_adjustment else 0,repeat_counts=[3,2] if bounded_adjustment else [3,3],
    wait_for_pid=154937,wait_for_script='${ARTIFACT_ROOT}/codex_local_qwen8b_morehop_generation_20260907_v1/study.py',
    methods=['finite','sac_mlp'],maximum_fresh_attributions=10 if bounded_adjustment else 12,memory_allowance_bytes=72500000,
    relative_l2_limit=0.001,flipped_mass_limit=0.0001,maximum_ratio_to_continue=0.99,
    torch_checkpoint_source_sha256='255ec2eccfa184eb3e13ef4791955d1f6f010d8ef1f59e7a8eccb9da26161314',
    purpose='One small engineering screen, only original NI2 and MH5 development cases, unchanged finite-P1 vs native MLP GEMM reuse through the unmodified public PyTorch SAC context API. Maximum12 fresh attributions (one warm+two measured each), including one candidate warm profiler. No new dataset, FT sweep, VJP, generation, quality curves or sign intervention. Early stop on failed correctness/memory boundary or no credible preliminary latency gain.',
    cache_provenance='Original root forward executes all native operations. Actual MLP inputs and weight versions are checked before replay reuses its original3 GEMM outputs. Distinct contexts per layer prevent reverse-order mismatches; allow_cache_entry_mutation=False. Native replay calls still counted, saved vs reused GEMMs explicitly distinguished. Actual model/FA/finite-P1 functions stay unchanged.',
    quality_cost_claim='Screen only. Same-rule improvement is provisional on two selected examples, not an FT win or independent quality result. Reference ordinary memory comes from the already verified same-model/input/target development record; fresh ordinary and seq-only FT comparison permitted only in separately frozen bounded followup.',
    repeats='At most2cases x2methods x3calls=12; candidate NI2 warm profile is inside the12, not an extra call. Gate evaluated after each pair, performance gate after the first case, stop early on failure.',
    budget={'maximum_root_forwards':12,'maximum_layer_replay_calls':432,'maximum_auxiliary_public_FA_calls':432,'maximum_finite_FA_calls':432,'new_quality_queries':0,'native_vjps':0})
if bounded_adjustment:
    prior_dir=A/'snapshot${ARTIFACT_ROOT}/codex_native_mlp_reuse_screen_20260907_v1'
    import zipfile
    with zipfile.ZipFile(prior_dir/'review_bundle.zip') as z:
        prior_raw=z.read('results.json');prior=json.loads(prior_raw)
    assert prior['status']=='screen_stopped_at_gate' and prior['fresh_attribution_calls']==2
    check=prior['records'][0]['checks'][0]
    assert check['full_signed_exact'] and not check['new_peak_below_memory_limit']
    p.update(wait_for_pid=155092,wait_for_script='${ARTIFACT_ROOT}/codex_native_mlp_reuse_screen_20260907_v1/study.py',
        previous_screen_raw_sha256=hashlib.sha256(prior_raw).hexdigest(),previous_screen_attributions=2,
        total_family_attribution_ceiling=12,
        adjustment_reason='The full36-layer official SAC cache passed exact same-job signed-vector/input checks but exceeded the existing memory ceiling by499332320bytes on NI2. Cache only last32MLPs: analytically removes650564? bytes of input plus3linear outputs at N1241; exact retained bytes are measured. One mechanically chosen reduction, no quality tuning or parameter grid.',
        repeats='Remaining budget10: NI2 one warm+two measured per method (6 calls); MH5 one warm+one measured per method (4 calls). Together with2 stopped full-cache calls, family maximum12. Warm candidate profile included. MH5 single measured pair is only a preliminary screen, not stable timing confirmation.',
        purpose='Bounded memory adjustment of the same native-PyTorch MLP reuse optimization. Last32 layers instead of all36; every other model/FA/finite operation unchanged. First screen stopped after2 actual calls on memory; this run permits at most10 additional calls, no new curves/FT sweep/generation/VJPs. Early stop retained.',
        budget={'maximum_root_forwards':10,'maximum_layer_replay_calls':360,'maximum_auxiliary_public_FA_calls':360,'maximum_finite_FA_calls':360,'new_quality_queries':0,'native_vjps':0})
    # NI2 has N1241, B2, FP16; each omitted MLP retained input D4096 plus
    # gate/up/down outputs D12288+D12288+D4096. Four omitted layers.
    saving=4*2*1241*2*(4096+12288+12288+4096)
    p['adjustment_predicted_retained_bytes_saving_NI2']=saving
    p['adjustment_reason']=p['adjustment_reason'].replace('650564?',str(saving))
protocol=A/f'native_mlp_reuse_screen{suffix}_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
remote='${ARTIFACT_ROOT}/codex_native_mlp_reuse_screen_20260907_v2' if bounded_adjustment else '${ARTIFACT_ROOT}/codex_native_mlp_reuse_screen_20260907_v1'
request_path=A/f'launch_native_mlp_reuse_screen{suffix}_20260907.json'
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),remote,
    f'study.py={study}',f'protocol.json={protocol}',*[f'{n}={A/n}' for n in p['sources']],
    '--request',str(request_path)],check=True)
request=json.loads(request_path.read_text())
assert len(request['cmd'])<128000
print(json.dumps({'study_sha256':sha(study),'protocol_sha256':sha(protocol),'maximum_attributions':p['maximum_fresh_attributions'],'new_quality_queries':0}))
