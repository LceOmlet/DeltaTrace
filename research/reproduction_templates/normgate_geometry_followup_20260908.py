"""Bounded CPU-only algebra on saved actual norm-gate captures.

Usage: python normgate_geometry_followup_20260908.py EXPERIMENT_DIRECTORY
Never invokes a model, GPU, native norm/attention, or a production pullback.
RMS/SiLU/Jacobians below are explicitly mathematical attribution factors, not
separately returned native internal tensors. Original inputs/results are read-only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MACA_VISIBLE_DEVICES']='-1'
import hashlib,json,signal,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F


STEPS=('1','10','20')
CPU_WALL_SECONDS=180


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def norm_value(x,w,eps):
    return w*x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)


def norm_jacobian_action(x,delta,w,eps):
    radius=(x.square().mean(-1,keepdim=True)+eps).sqrt()
    return w*(delta/radius-x*(x*delta).sum(-1,keepdim=True)/(x.shape[-1]*radius.pow(3)))


def norm_finite_action(x0,x1,delta,w,eps):
    r0=(x0.square().mean(-1,keepdim=True)+eps).sqrt()
    r1=(x1.square().mean(-1,keepdim=True)+eps).sqrt()
    mid=(x0+x1)*.5
    return w*((1/r0+1/r1)*.5*delta-2*mid*(mid*delta).sum(-1,keepdim=True)
              /(x0.shape[-1]*r0*r1*(r0+r1)))


def silu_secant(z0,z1):
    delta=z1-z0;nonzero=delta!=0;s0=F.silu(z0);s1=F.silu(z1)
    sigmoid=z0.sigmoid();derivative=sigmoid*(1+z0*(1-sigmoid))
    return torch.where(nonzero,(s1-s0)/torch.where(nonzero,delta,torch.ones_like(delta)),derivative)


def cpu_only(value):
    if isinstance(value,torch.Tensor):assert value.device.type=='cpu'
    elif isinstance(value,dict):
        for child in value.values():cpu_only(child)


def decompose_case(case,captures):
    cpu_only(captures)
    paired=captures['paired'];points=captures['points'];coeff=captures['coefficients']
    assert set(points)=={'0','1','10','20'}
    B,T,H,D=points['0']['e']['o'].shape
    assert B==1 and paired['e']['o'].shape==(2,T,H,D)
    assert captures['weight'].shape==(D,) and captures['eps']>0
    w=captures['weight'].double();eps=float(captures['eps'])
    m_all=coeff['mnorm'].reshape(B,T,H,D)
    assert coeff['mo_before_cast'].shape==coeff['mz'].shape==(B,T,H,D)
    P=case['input']['prompt_length'];keep=set(case['input']['keep'])
    assert case['input']['total_length']==T and all(0<=i<P for i in keep)
    names=[
        'RMS_reference_effect_from_actual_o','current_theoretical_RMS_prediction','current_actual_mo_prediction',
        'J1_RMS_prediction','Lstar_RMS_prediction',
        'current_RMS_error','J1_conditional_curvature','L_minus_J1_parallel','L_minus_J1_perpendicular',
        'Jclean_conditional_curvature','J1_minus_Jclean_transfer',
        'Lstar_RMS_error','Lstar_minus_current_prediction','Lstar_minus_current_perpendicular_prediction',
        'current_actual_normgate_error','current_theoretical_normgate_error',
        'actual_minus_theoretical_current_prediction','symmetric_theoretical_normgate_error',
        'symmetric_minus_current_full_prediction','symmetric_RMS_factor_prediction_change',
        'symmetric_SiLU_factor_prediction_change','symmetric_product_only_prediction_change',
        'symmetric_finite_curvature_factor_prediction_change']
    output={}
    for step in STEPS:
        start=time.perf_counter();deleted=set(case['scoring_points'][step]['input_receipt']['deleted_positions'])
        assert deleted<=keep
        groups={'eligible_deleted':sorted(deleted),'eligible_kept':sorted(keep-deleted),
                'other_prompt':sorted(set(range(P))-keep),'fixed_response':list(range(P,T))}
        assert sum(map(len,groups.values()))==T and set().union(*(set(v) for v in groups.values()))==set(range(T))
        totals={name:torch.zeros((B,T),dtype=torch.float64,device='cpu') for name in names};heads=[]
        max_checks={name:0. for name in ['RMS_three_term_closure','Jclean_transfer_closure','Lstar_endpoint_conservation',
            'current_endpoint_conservation','Lstar_change_only_perpendicular','symmetric_factor_closure',
            'symmetric_curvature_factor_closure','symmetric_error_change_closure']}
        zeros={'zero_full_endpoint_delta':0,'zero_endpoint_with_nonzero_condition':0,'zero_condition_delta':0}
        for h in range(H):
            o0=paired['e']['o'][0::2,:,h].double();o1=paired['e']['o'][1::2,:,h].double()
            oc=points['0']['e']['o'][:,:,h].double();oa=points[step]['e']['o'][:,:,h].double()
            z0=paired['c']['z'][0::2,:,h].double();z1=paired['c']['z'][1::2,:,h].double()
            zc=points['0']['c']['z'][:,:,h].double();za=points[step]['c']['z'][:,:,h].double()
            yc=points['0']['c']['norm_output'].reshape(B,T,H,D)[:,:,h].double()
            ya=points[step]['c']['norm_output'].reshape(B,T,H,D)[:,:,h].double()
            m=m_all[:,:,h].double();mo=coeff['mo_before_cast'][:,:,h].double();mz=coeff['mz'][:,:,h].double()
            n0=norm_value(o0,w,eps);n1=norm_value(o1,w,eps);nc=norm_value(oc,w,eps);na=norm_value(oa,w,eps)
            s0=F.silu(z0);s1=F.silu(z1);sc=F.silu(zc);sa=F.silu(za)
            full=o1-o0;delta=oc-oa;dn=nc-na;dz=zc-za;ds=sc-sa
            full_sq=full.square().sum(-1);condition_sq=delta.square().sum(-1);valid=full_sq>0
            alpha=torch.zeros_like(full_sq);alpha[valid]=(full*delta).sum(-1)[valid]/full_sq[valid]
            parallel=alpha[...,None]*full;perpendicular=delta-parallel
            zeros['zero_full_endpoint_delta']+=int((~valid).sum())
            zeros['zero_endpoint_with_nonzero_condition']+=int(((~valid)&(condition_sq>0)).sum())
            zeros['zero_condition_delta']+=int((condition_sq==0).sum())
            # Zero endpoint difference spans {0}; its projection is exactly 0.
            # For this deterministic theoretical RMS, Delta f must also be 0.
            assert bool((n1-n0)[~valid].eq(0).all())
            L=lambda v:norm_finite_action(o0,o1,v,w,eps)
            J1=lambda v:norm_jacobian_action(o1,v,w,eps)
            Jc=lambda v:norm_jacobian_action(oc,v,w,eps)
            Ldelta=L(delta);Jdelta=J1(delta);Jcdelta=Jc(delta)
            residual=n1-n0-J1(full)
            def Lstar(v):
                projection=torch.zeros_like(full_sq)
                projection[valid]=(full*v).sum(-1)[valid]/full_sq[valid]
                return J1(v)+residual*projection[...,None]
            star_delta=Lstar(delta);v=m*s1;reduce=lambda x:x.sum(-1)
            hs=silu_secant(z0,z1)*dz
            native_effect=reduce(m*(yc-ya))
            actual_prediction=reduce(mo*delta+mz*dz)
            theoretical_prediction=reduce(m*(s1*Ldelta+n0*hs))
            symmetric_prediction=reduce(m*((s0+s1)*.5*Ldelta+(n0+n1)*.5*hs))
            terms={
                'RMS_reference_effect_from_actual_o':reduce(v*dn),
                'current_theoretical_RMS_prediction':reduce(v*Ldelta),
                'current_actual_mo_prediction':reduce(mo*delta),
                'J1_RMS_prediction':reduce(v*Jdelta),
                'Lstar_RMS_prediction':reduce(v*star_delta),
                'current_RMS_error':reduce(v*(Ldelta-dn)),
                'J1_conditional_curvature':reduce(v*(Jdelta-dn)),
                'L_minus_J1_parallel':reduce(v*(L(parallel)-J1(parallel))),
                'L_minus_J1_perpendicular':reduce(v*(L(perpendicular)-J1(perpendicular))),
                'Jclean_conditional_curvature':reduce(v*(Jcdelta-dn)),
                'J1_minus_Jclean_transfer':reduce(v*(Jdelta-Jcdelta)),
                'Lstar_RMS_error':reduce(v*(star_delta-dn)),
                'Lstar_minus_current_prediction':reduce(v*(star_delta-Ldelta)),
                'Lstar_minus_current_perpendicular_prediction':reduce(v*(Lstar(perpendicular)-L(perpendicular))),
                'current_actual_normgate_error':actual_prediction-native_effect,
                'current_theoretical_normgate_error':theoretical_prediction-native_effect,
                'actual_minus_theoretical_current_prediction':actual_prediction-theoretical_prediction,
                'symmetric_theoretical_normgate_error':symmetric_prediction-native_effect,
                'symmetric_minus_current_full_prediction':symmetric_prediction-theoretical_prediction,
                'symmetric_RMS_factor_prediction_change':reduce(-.5*m*(s1-s0)*Ldelta),
                'symmetric_SiLU_factor_prediction_change':reduce(.5*m*(n1-n0)*hs),
                'symmetric_product_only_prediction_change':reduce(.5*m*((n1-n0)*ds-(s1-s0)*dn)),
                'symmetric_finite_curvature_factor_prediction_change':reduce(.5*m*((n1-n0)*(hs-ds)-(s1-s0)*(Ldelta-dn)))}
            checks={
                'RMS_three_term_closure':terms['current_RMS_error']-terms['J1_conditional_curvature']-terms['L_minus_J1_parallel']-terms['L_minus_J1_perpendicular'],
                'Jclean_transfer_closure':terms['J1_conditional_curvature']-terms['Jclean_conditional_curvature']-terms['J1_minus_Jclean_transfer'],
                'Lstar_endpoint_conservation':Lstar(full)-(n1-n0),
                'current_endpoint_conservation':L(full)-(n1-n0),
                'Lstar_change_only_perpendicular':terms['Lstar_minus_current_prediction']-terms['Lstar_minus_current_perpendicular_prediction'],
                'symmetric_factor_closure':terms['symmetric_minus_current_full_prediction']-terms['symmetric_RMS_factor_prediction_change']-terms['symmetric_SiLU_factor_prediction_change'],
                'symmetric_curvature_factor_closure':terms['symmetric_minus_current_full_prediction']-terms['symmetric_product_only_prediction_change']-terms['symmetric_finite_curvature_factor_prediction_change'],
                'symmetric_error_change_closure':terms['symmetric_theoretical_normgate_error']-terms['current_theoretical_normgate_error']-terms['symmetric_minus_current_full_prediction']}
            for name,value in checks.items():
                maximum=float(value.abs().max());max_checks[name]=max(max_checks[name],maximum)
                assert maximum<1e-7,(step,h,name,maximum)
            for name,value in terms.items():
                assert bool(torch.isfinite(value).all()),(step,h,name)
                totals[name]+=value
            heads.append({'head':h,'terms':{name:float(value.sum()) for name,value in terms.items()}})
        values={name:float(value.sum()) for name,value in totals.items()}
        total_checks={
            'RMS_three_term_closure':values['current_RMS_error']-values['J1_conditional_curvature']-values['L_minus_J1_parallel']-values['L_minus_J1_perpendicular'],
            'Jclean_transfer_closure':values['J1_conditional_curvature']-values['Jclean_conditional_curvature']-values['J1_minus_Jclean_transfer'],
            'Lstar_change_only_perpendicular':values['Lstar_minus_current_prediction']-values['Lstar_minus_current_perpendicular_prediction'],
            'symmetric_factor_closure':values['symmetric_minus_current_full_prediction']-values['symmetric_RMS_factor_prediction_change']-values['symmetric_SiLU_factor_prediction_change'],
            'symmetric_curvature_factor_closure':values['symmetric_minus_current_full_prediction']-values['symmetric_product_only_prediction_change']-values['symmetric_finite_curvature_factor_prediction_change']}
        assert all(abs(v)<1e-7 for v in total_checks.values()),total_checks
        original=case['scoring_points'][step]['normgate_decomposition']
        checks_against_saved={
            'current_RMS_error':values['current_RMS_error']-original['terms']['RMS_conditional_B2_operator'],
            'current_actual_normgate_error':values['current_actual_normgate_error']-original['measured_prediction_minus_actual']}
        assert all(abs(v)<1e-7 for v in checks_against_saved.values()),checks_against_saved
        output[step]={'terms':values,'per_head':heads,'token_groups':{group:{name:float(value[0,indices].sum())
            for name,value in totals.items()} for group,indices in groups.items()},'token_group_sizes':{k:len(v) for k,v in groups.items()},
            'zero_direction_counts':zeros,'max_abs_checks':max_checks,'total_closure_checks':total_checks,'difference_from_saved_decomposition':checks_against_saved,
            'CPU_algebra_seconds':time.perf_counter()-start}
    return output


def main():
    assert len(sys.argv)==2,'Supply the original experiment directory only.'
    directory=Path(sys.argv[1]).resolve();source=directory/'results.json';out=directory/'normgate_geometry_followup.json'
    assert source.is_file() and not out.exists(),'Original results must exist; refuse to overwrite a prior followup.'
    start=time.perf_counter();r={'status':'starting','source_results_sha256':digest(source),'self_sha256':digest(Path(__file__)),
        'sign_convention':'prediction_minus_actual','fixed_steps':list(STEPS),'CPU_wall_budget_seconds':CPU_WALL_SECONDS,
        'model_calls':0,'GPU_calls':0,'native_model_operator_calls':0,'parameter_searches':0,'cases':{},'skipped_cases':{}}
    def save():
        temporary=out.with_suffix('.partial');temporary.write_text(json.dumps(r,indent=2));temporary.replace(out)
    def timeout(*_args):raise TimeoutError('Frozen CPU algebra wall budget exceeded.')
    try:
        if hasattr(signal,'SIGALRM'):signal.signal(signal.SIGALRM,timeout);signal.alarm(CPU_WALL_SECONDS)
        torch.set_num_threads(4);torch.set_default_device('cpu')
        original=json.loads(source.read_bytes())
        r['source_experiment_status']=original.get('status')
        for key,case in original['cases'].items():
            receipt=case.get('private_normgate_input_artifact')
            if receipt is None:
                r['skipped_cases'][key]='No completed actual norm-gate capture artifact; no rerun or reconstruction.';continue
            path=(directory/receipt['file']).resolve()
            assert path.parent==directory and path.is_file() and digest(path)==receipt['sha256']
            assert case['parent_vector_relative_L2']<=0.02
            captures=torch.load(path,map_location='cpu',weights_only=True)
            r['cases'][key]={'input_artifact':receipt,'input_artifact_sha256_before':digest(path)};save()
            r['cases'][key]['steps']=decompose_case(case,captures)
            r['cases'][key]['input_artifact_sha256_after']=digest(path)
            assert r['cases'][key]['input_artifact_sha256_before']==r['cases'][key]['input_artifact_sha256_after']
            save();del captures
        assert r['cases'],'No completed case available for the fixed followup.'
        assert digest(source)==r['source_results_sha256'],'Original result file changed during read-only analysis.'
        r['status']='CPU_geometry_followup_complete'
    except Exception:
        r['status']='failed';r['error']=traceback.format_exc()
    finally:
        if hasattr(signal,'SIGALRM'):signal.alarm(0)
        r['CPU_job_seconds']=time.perf_counter()-start
        r['scope']='Three fixed algebra diagnostics on actual saved operands. No new production execution, full input attribution, candidate ranking, RISE/MAS, or repair guarantee.'
        r['precision']='Saved production coefficients unchanged. CPU64 theoretical RMS/SiLU factors and Jacobian actions only; native fused output used solely as the observed effect.'
        r['RMS_reference_contract']='RMS reference effect evaluates the mathematical RMS on actual captured o_clean/o_A with the original weight/eps and frozen m*s1. It is not a separately returned native RMS tensor.'
        r['Lstar_contract']='J1 at B2 input plus minimum-Frobenius rank-one endpoint correction. Zero Delta uses J1; zero endpoint span projection is 0. Changed prediction equals its perpendicular component.'
        r['symmetric_contract']='Only theoretical norm-times-SiLU allocation changes, with existing finite RMS/SiLU maps. Full RMS/Silu multiplier changes included, not merely the true n*s product residual.'
        save();print(json.dumps({'status':r['status'],'CPU_seconds':r['CPU_job_seconds'],'cases':list(r['cases']),'error':r.get('error')}),flush=True)
    return 0 if r['status']=='CPU_geometry_followup_complete' else 1


if __name__=='__main__':raise SystemExit(main())
