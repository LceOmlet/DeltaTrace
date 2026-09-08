"""Frozen CPU algebra: can either meaningful product order touch MLP6 error?

No candidate model/attribution, GPU, GEMM, weight read or lambda search. The
saved symmetric multipliers and actual scoring features remain the reference.
"""
import gc,hashlib,json,os,signal,time,traceback,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
started=time.perf_counter();r={'status':'starting','protocol':p,'GPU_calls':0,'GEMM_calls':0,'model_calls':0,
    'DT_calls':0,'FT_calls':0,'score_calls':0,'weight_reads':0,'points':{}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2,allow_nan=False));q.replace(A/'results.json')


def stats(v,groups):
    assert v.device.type=='cpu' and v.ndim==3 and v.shape[:2]==(1,T) and bool(v.isfinite().all())
    def summary(x):return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),
        'negative':float(x.clamp_max(0).sum()),'absolute_sum':float(x.abs().sum())}
    return {**summary(v),'per_token':v.sum(-1)[0].tolist(),'groups':{k:summary(v[:,idx]) for k,idx in groups.items()}}


def grouped(deleted):
    eligible=set(info['keep']);selected=set(deleted);assert selected<=eligible
    out={'eligible_deleted':sorted(selected),'eligible_kept':sorted(eligible-selected),
        'other_prompt':sorted(set(range(info['prompt_length']))-eligible),
        'fixed_response':list(range(info['prompt_length'],T))}
    assert sorted(i for ids in out.values() for i in ids)==list(range(T));return out


def point(label,clean,deleted,groups,previous):
    dg=clean['gate_output'].double()-deleted['gate_output'].double()
    du=clean['up_output'].double()-deleted['up_output'].double()
    det=rho*(ds*du-du_pair*sigma*dg)
    sym_rounding=(mu_sym-mu_actual)*du+(mg_sym-mg_actual)*dg
    theory_shift={'content1':det*0.5,'reverse':-det*0.5}
    # Compare new CPU64 ideal endpoint coefficients to the ACTUAL saved
    # symmetric coefficients. This includes their small rounding difference.
    explicit={'content1':(rho*s1-mu_actual)*du+(rho*u0*sigma-mg_actual)*dg,
        'reverse':(rho*s0-mu_actual)*du+(rho*u1*sigma-mg_actual)*dg}
    out={'sign_convention':'prediction_minus_actual','actual_current_MLP_error':previous['saved_prediction_minus_actual'],
        'actual_current_input_prediction':previous['saved_input_prediction'],
        'actual_current_output_effect':previous['actual_output_effect'],
        'actual_current_interaction_error':previous['terms']['bilinear_content_gate_interaction']['total'],
        'actual_current_SiLU_curvature_error':previous['terms']['SiLU_scalar_secant_curvature']['total'],
        'determinant':stats(det,groups),'symmetric_theory_minus_actual_coefficients':stats(sym_rounding,groups),'orders':{}}
    magnitude=(rho*ds*du).abs()+(rho*du_pair*sigma*dg).abs()
    out['determinant_cancellation']={'absolute_determinant':float(det.abs().sum()),
        'absolute_two_terms':float(magnitude.sum()),
        'ratio':float(det.abs().sum()/magnitude.sum()) if float(magnitude.sum()) else None,
        'scope':'Weighted channelwise transverse sensitivity; a correlation diagnostic, not a proof of global fit.'}
    for rule in ['content1','reverse']:
        residue=explicit[rule]-theory_shift[rule]-sym_rounding
        err=previous['saved_prediction_minus_actual']+float(explicit[rule].sum())
        out['orders'][rule]={'theoretical_shift_from_exact_symmetric':stats(theory_shift[rule],groups),
            'shift_from_actual_saved_branch_coefficients':stats(explicit[rule],groups),
            'analytic_MLP_error_if_input_GEMM_transfer_unchanged':err,
            'analytic_absolute_error_reduction_if_transfer_unchanged':abs(previous['saved_prediction_minus_actual'])-abs(err),
            'analytic_input_prediction_if_transfer_unchanged':previous['saved_input_prediction']+float(explicit[rule].sum()),
            'max_channel_identity_residual':float(residue.abs().max()),'total_identity_residual':float(residue.sum()),
            'group_error_if_input_GEMM_transfer_unchanged':{k:previous['group_saved_error'][k]+float(explicit[rule][:,idx].sum()) for k,idx in groups.items()}}
        r['points'][label]=out;save()
        assert float(residue.abs().max())<1e-7 and abs(float(residue.sum()))<1e-7
    out['numerical_scope']='Only an analytic branch-boundary change. New native BF16 casts/input GEMMs are NOT executed; their changed transfer and all earlier-layer redistribution remain unmeasured. No new MAS/RISE/needle.'
    r['points'][label]=out;save()


try:
    def timeout(*_):raise TimeoutError('Frozen CPU-only reachability budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds'])
    for name,h in p['files_sha256'].items():assert sha(A/name)==h
    r['input_hashes_before']={name:sha(item['path']) for name,item in p['inputs'].items()}
    assert all(r['input_hashes_before'][name]==item['sha256'] for name,item in p['inputs'].items())
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    actual=json.loads(Path(p['inputs']['actual_results']['path']).read_bytes())
    previous=json.loads(Path(p['inputs']['mechanism_results']['path']).read_bytes())
    assert actual['status']=='decoder19_6_actual_conditional_observation_complete'
    assert previous['status']=='MLP6_actual_mechanism_diagnostic_complete' and previous['saved_vs_recomputed_mnorm']['bitwise_equal']
    info=actual['input'];T=info['total_length'];assert p['fixed_steps']==['1','10','20'] and p['layer_index']==6
    artifact=torch.load(p['inputs']['actual_private']['path'],map_location='cpu',weights_only=True)
    pair={k:artifact['layers']['6']['paired']['d'][k] for k in ['gate_output','up_output','silu_output']}
    points={step:{k:artifact['scoring'][step]['6']['d'][k] for k in ['gate_output','up_output']} for step in ['0','1','10','20']}
    saved_mnorm=artifact['layers']['6']['coeff']['m_mlp_norm_output']
    del artifact;gc.collect()
    diagnostic=torch.load(p['inputs']['recomputed_coefficients']['path'],map_location='cpu',weights_only=True)
    coeff=diagnostic['coefficients'];assert torch.equal(coeff['mnorm'],saved_mnorm)
    rho=coeff['m_product'].double();mu_actual=coeff['mu'].double();mg_actual=coeff['mg'].double()
    del diagnostic,coeff,saved_mnorm;gc.collect()
    g0,g1=pair['gate_output'][0::2].double(),pair['gate_output'][1::2].double()
    u0,u1=pair['up_output'][0::2].double(),pair['up_output'][1::2].double()
    s0,s1=pair['silu_output'][0::2].double(),pair['silu_output'][1::2].double()
    assert all(x.shape==rho.shape for x in [g0,g1,u0,u1,s0,s1,mu_actual,mg_actual]) and rho.shape[:2]==(1,T)
    dg_pair=g1-g0;du_pair=u1-u0;ds=s1-s0;sigmoid=g0.sigmoid();derivative=sigmoid*(1+g0*(1-sigmoid))
    sigma=torch.where(dg_pair!=0,ds/torch.where(dg_pair!=0,dg_pair,torch.ones_like(dg_pair)),derivative)
    mu_sym=rho*(s0+s1)*0.5;mg_sym=rho*(u0+u1)*0.5*sigma
    r['zero_gate_delta']={'channels':int((dg_pair==0).sum()),'native_silu_delta_nonzero':int(((dg_pair==0)&(ds!=0)).sum()),
        'policy':'Use original SiLU secant derivative fallback; no arbitrary epsilon, division recovery, or zero-channel removal.'}
    r['status']='four_fixed_CPU_algebra_points';save()
    point('B2',{k:v[1::2] for k,v in pair.items()},{k:v[0::2] for k,v in pair.items()},grouped(info['keep']),previous['points']['B2'])
    for step in p['fixed_steps']:
        point(step,points['0'],points[step],grouped(actual['points'][step]['input_receipt']['deleted_positions']),previous['points'][step])
    r['predefined_direction_checks']={rule:{'analytic_absolute_error_reduced_at_both_early_and_mid':all(
        r['points'][step]['orders'][rule]['analytic_absolute_error_reduction_if_transfer_unchanged']>0 for step in ['1','10']),
        'B2_endpoint_analytic_shift':r['points']['B2']['orders'][rule]['shift_from_actual_saved_branch_coefficients']['net'],
        'B1_allEOS_analytic_shift':r['points']['20']['orders'][rule]['shift_from_actual_saved_branch_coefficients']['net']}
        for rule in ['content1','reverse']}
    r['input_hashes_after']={name:sha(item['path']) for name,item in p['inputs'].items()}
    assert r['input_hashes_after']==r['input_hashes_before'] and not torch.cuda.is_initialized()
    r['status']='MLP6_CPU_order_reachability_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['CPU_job_seconds']=time.perf_counter()-started
    r['interpretation']='Two predefined functional endpoint orders, not a lambda scan or fitted winner. A favorable local prediction only licenses a bounded actual layer6 check. Fixed endpoint-conserving linear allocations cannot repair nonlinear effects along the complete endpoint direction.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*p['files_sha256'],'protocol.json','results.json']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'CPU_seconds':r['CPU_job_seconds'],'error':r.get('error')}),flush=True)
