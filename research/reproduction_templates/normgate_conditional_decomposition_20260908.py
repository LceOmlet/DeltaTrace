"""CPU audit of one native GDN norm-gate conditional effect.

Input activations and coefficients must be actual CPU captures. RMS and SiLU
below are explicitly FP64 attribution algebra, not native intermediate outputs
or an alternate model forward. No production coefficient or operator is changed.
Memory is linear in tokens; process one head at a time. Capture/copy/CPU costs
belong to diagnostics and must not be advertised as zero-cost model execution.
"""
import math
import time
import torch
import torch.nn.functional as F


def _rms(x,weight,eps):
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight


def _rms_finite_forward(o0,o1,delta,weight,eps):
    """Apply the declared symmetric finite RMS operator, without a matrix."""
    r0=(o0.square().mean(-1,keepdim=True)+eps).sqrt()
    r1=(o1.square().mean(-1,keepdim=True)+eps).sqrt()
    mean=(o0+o1)*.5
    inv_mean=(1/r0+1/r1)*.5
    rank_factor=-2/(o0.shape[-1]*r0*r1*(r0+r1))
    return weight*(inv_mean*delta+rank_factor*mean*(mean*delta).sum(-1,keepdim=True))


def _silu_secant(z0,z1,s0,s1):
    delta=z1-z0;nonzero=delta!=0
    sigmoid=z0.sigmoid();derivative=sigmoid*(1+z0*(1-sigmoid))
    return torch.where(nonzero,(s1-s0)/torch.where(nonzero,delta,torch.ones_like(delta)),derivative)


def _geometry(o0,o1,oc,oa,eps,rms_token_error):
    """Small per-head associations, not a correctness or repair certificate."""
    def stats(x):
        x=x.flatten()
        if not x.numel():return {'count':0,'min':None,'median':None,'max':None}
        return {'count':x.numel(),'min':float(x.min()),'median':float(x.quantile(.5)),'max':float(x.max())}
    radii={k:(x.square().mean(-1)+eps).sqrt() for k,x in [('r0',o0),('r1',o1),('r_clean',oc),('r_A',oa)]}
    ratio=radii['r1']/radii['r0'];endpoint=o1-o0;condition=oc-oa
    endpoint_sq=endpoint.square().sum(-1);condition_sq=condition.square().sum(-1)
    endpoint_valid=endpoint_sq>0;condition_valid=condition_sq>0;valid=endpoint_valid&condition_valid
    alpha=(endpoint*condition).sum(-1)[endpoint_valid]/endpoint_sq[endpoint_valid]
    residual=condition[valid]-((endpoint[valid]*condition[valid]).sum(-1)/endpoint_sq[valid])[:,None]*endpoint[valid]
    perp_fraction=residual.square().sum(-1).sqrt()/condition_sq[valid].sqrt()
    weights=rms_token_error.abs()
    def weighted(value,weight):
        total=float(weight.sum())
        return {'absolute_RMS_error_weight_sum':total,'weighted_mean':float((value*weight).sum())/total if total else None}
    return {'radius_summaries':{k:stats(v) for k,v in radii.items()},'r1_over_r0':stats(ratio),
            'parallel_coefficient':stats(alpha),'perpendicular_fraction_of_condition_norm':stats(perp_fraction),
            'zero_endpoint_delta_count':int((~endpoint_valid).sum()),'zero_condition_delta_count':int((~condition_valid).sum()),
            'zero_endpoint_with_nonzero_condition_count':int(((~endpoint_valid)&condition_valid).sum()),
            'absolute_RMS_error_weighted_r1_over_r0':weighted(ratio,weights),
            'absolute_RMS_error_weighted_perpendicular_fraction':weighted(perp_fraction,weights[valid]),
            'direction_contract':'parallel alpha=<Delta_o,delta_o>/||Delta_o||^2; perpendicular ratio is ||delta_o-alpha*Delta_o||/||delta_o||. Zero denominators excluded and counted.'}


def decompose_normgate(paired,clean,current,coefficients,weight,eps,b1_eos=None):
    """Prediction-minus-actual decomposition using frozen B2 coefficients.

    paired/clean/current use the focused driver's {'c': ..., 'e': ...} feature
    dictionaries. paired rows are [EOS0,input0,...]; clean/current are actual B1
    scoring captures. Only e['o'], c['z'], c['norm_output'] are consumed.
    coefficients needs actual mnorm, mo_before_cast and mz. weight is the
    unchanged native module.norm.weight copied to CPU; eps is module.norm.eps.

    Optional b1_eos is the actual scorer all-EOS capture. Supplying it splits
    product_B2_baseline into a B1-baseline product and explicit baseline transfer.
    Without it, the combined product is retained and clearly marked; no missing
    native baseline is reconstructed. The top-level sum always uses the B2
    product term, never both this term and its optional two-term subdivision.
    """
    started=time.perf_counter()
    required=lambda x:[x['e']['o'],x['c']['z'],x['c']['norm_output']]
    for capture in [paired,clean,current]+([] if b1_eos is None else [b1_eos]):
        assert all(isinstance(x,torch.Tensor) and x.device.type=='cpu' for x in required(capture))
    assert weight.device.type=='cpu' and weight.ndim==1 and eps>0
    assert all(coefficients[k].device.type=='cpu' for k in ['mnorm','mo_before_cast','mz'])
    paired_o=paired['e']['o'];B,T,H,D=clean['e']['o'].shape
    assert paired_o.shape==(2*B,T,H,D) and current['e']['o'].shape==(B,T,H,D)
    assert weight.shape==(D,)
    for capture in [paired,clean,current]+([] if b1_eos is None else [b1_eos]):
        assert capture['c']['z'].shape==capture['e']['o'].shape
        assert capture['c']['norm_output'].shape==(*capture['e']['o'].shape[:2],H*D)
    assert coefficients['mnorm'].shape==(B,T,H*D)
    assert coefficients['mo_before_cast'].shape==coefficients['mz'].shape==(B,T,H,D)
    if b1_eos is not None:assert b1_eos['e']['o'].shape==(B,T,H,D)
    names=['RMS_conditional_B2_operator','SiLU_conditional_B2_operator',
           'product_B2_baseline','B2_input_gate_factor_minus_B1_clean',
           'actual_mo_minus_theoretical_coefficient','actual_mz_minus_theoretical_coefficient',
           'theoretical_product_minus_native_fused_output']
    per_token={name:torch.zeros((B,T),dtype=torch.float64,device='cpu') for name in names}
    subdivisions={name:torch.zeros((B,T),dtype=torch.float64,device='cpu') for name in
        (['product_B1_EOS_baseline','B2_n0_minus_B1_EOS_factor'] if b1_eos is not None else [])}
    operator_subdivisions={name:torch.zeros((B,T),dtype=torch.float64,device='cpu') for name in
        (['RMS_B1_endpoint_operator','RMS_B2_minus_B1_operator','SiLU_B1_endpoint_operator','SiLU_B2_minus_B1_operator']
         if b1_eos is not None else [])}
    measured=torch.zeros((B,T),dtype=torch.float64,device='cpu');prediction=measured.clone();actual=measured.clone()
    metrics={};per_head=[];w=weight.double();m_all=coefficients['mnorm'].reshape(B,T,H,D)
    def discrepancy(name,left,right):
        row=metrics.setdefault(name,{'difference_squared_sum':0.,'reference_squared_sum':0.,'max_abs_difference':0.})
        diff=left-right;row['difference_squared_sum']+=float(diff.square().sum())
        row['reference_squared_sum']+=float(right.square().sum())
        row['max_abs_difference']=max(row['max_abs_difference'],float(diff.abs().max()))
    for h in range(H):
        o0=paired_o[0::2,:,h].double();o1=paired_o[1::2,:,h].double()
        z0=paired['c']['z'][0::2,:,h].double();z1=paired['c']['z'][1::2,:,h].double()
        oc=clean['e']['o'][:,:,h].double();oa=current['e']['o'][:,:,h].double()
        zc=clean['c']['z'][:,:,h].double();za=current['c']['z'][:,:,h].double()
        yc=clean['c']['norm_output'].reshape(B,T,H,D)[:,:,h].double()
        ya=current['c']['norm_output'].reshape(B,T,H,D)[:,:,h].double()
        y1=paired['c']['norm_output'].reshape(2*B,T,H,D)[1::2,:,h].double()
        m=m_all[:,:,h].double();mo=coefficients['mo_before_cast'][:,:,h].double();mz=coefficients['mz'][:,:,h].double()
        n0=_rms(o0,w,eps);n1=_rms(o1,w,eps);nc=_rms(oc,w,eps);na=_rms(oa,w,eps)
        s0=F.silu(z0);s1=F.silu(z1);sc=F.silu(zc);sa=F.silu(za)
        do=oc-oa;dz=zc-za;dn=nc-na;ds=sc-sa
        dn_hat=_rms_finite_forward(o0,o1,do,w,eps)
        ds_hat=_silu_secant(z0,z1,s0,s1)*dz
        reduce=lambda x:x.sum(-1)
        actual_mo=reduce(mo*do);actual_mz=reduce(mz*dz)
        ideal_mo=reduce(m*s1*dn_hat);ideal_mz=reduce(m*n0*ds_hat)
        terms={
            'RMS_conditional_B2_operator':reduce(m*s1*(dn_hat-dn)),
            'SiLU_conditional_B2_operator':reduce(m*n0*(ds_hat-ds)),
            'product_B2_baseline':reduce(m*(n0-na)*ds),
            'B2_input_gate_factor_minus_B1_clean':reduce(m*(s1-sc)*dn),
            'actual_mo_minus_theoretical_coefficient':actual_mo-ideal_mo,
            'actual_mz_minus_theoretical_coefficient':actual_mz-ideal_mz,
            'theoretical_product_minus_native_fused_output':reduce(m*((nc*sc-na*sa)-(yc-ya)))}
        head_prediction=actual_mo+actual_mz;head_actual=reduce(m*(yc-ya));head_error=head_prediction-head_actual
        for name,value in terms.items():per_token[name]+=value
        prediction+=head_prediction;actual+=head_actual;measured+=head_error
        per_head.append({'head':h,'measured_prediction_minus_actual':float(head_error.sum()),
                         'terms':{name:float(value.sum()) for name,value in terms.items()},
                         'max_abs_token_closure_error':float((sum(terms.values())-head_error).abs().max()),
                         'RMS_geometry':_geometry(o0,o1,oc,oa,eps,terms['RMS_conditional_B2_operator'])})
        for name,left,right in [('o_clean_vs_B2_input',oc,o1),('z_clean_vs_B2_input',zc,z1),
                                ('theoretical_n_clean_vs_B2_input',nc,n1),('theoretical_s_clean_vs_B2_input',sc,s1),
                                ('native_fused_clean_vs_B2_input',yc,y1)]:discrepancy(name,left,right)
        if b1_eos is not None:
            oe=b1_eos['e']['o'][:,:,h].double();ze=b1_eos['c']['z'][:,:,h].double()
            ne=_rms(oe,w,eps);se=F.silu(ze)
            subdivisions['product_B1_EOS_baseline']+=reduce(m*(ne-na)*ds)
            subdivisions['B2_n0_minus_B1_EOS_factor']+=reduce(m*(n0-ne)*ds)
            # Reanchoring is a diagnostic algebra subdivision, never a new
            # production operator. Keep the same B2 factors s1 and n0 here.
            dn_hat_B1=_rms_finite_forward(oe,oc,do,w,eps)
            ds_hat_B1=_silu_secant(ze,zc,se,sc)*dz
            operator_subdivisions['RMS_B1_endpoint_operator']+=reduce(m*s1*(dn_hat_B1-dn))
            operator_subdivisions['RMS_B2_minus_B1_operator']+=reduce(m*s1*(dn_hat-dn_hat_B1))
            operator_subdivisions['SiLU_B1_endpoint_operator']+=reduce(m*n0*(ds_hat_B1-ds))
            operator_subdivisions['SiLU_B2_minus_B1_operator']+=reduce(m*n0*(ds_hat-ds_hat_B1))
            for name,left,right in [('o_B1_EOS_vs_B2_EOS',oe,o0),('z_B1_EOS_vs_B2_EOS',ze,z0),
                                    ('theoretical_n_B1_EOS_vs_B2_EOS',ne,n0),('theoretical_s_B1_EOS_vs_B2_EOS',se,s0)]:
                discrepancy(name,left,right)
    closure=sum(per_token.values())-measured
    assert float(closure.abs().max())<1e-7 and abs(float(closure.sum()))<1e-7
    if subdivisions:
        assert float((sum(subdivisions.values())-per_token['product_B2_baseline']).abs().max())<1e-7
        for name in ['RMS','SiLU']:
            assert float((operator_subdivisions[name+'_B1_endpoint_operator']+operator_subdivisions[name+'_B2_minus_B1_operator']
                          -per_token[name+'_conditional_B2_operator']).abs().max())<1e-7
    for row in metrics.values():
        row['relative_L2']=math.sqrt(row['difference_squared_sum']/max(row['reference_squared_sum'],1e-300))
    return {'sign_convention':'prediction_minus_actual; opposite of previous focused actual_minus_predicted',
            'algebra':'CPU FP64 RMS/SiLU factors from actual captured operands and original weight/eps; not native internal tensors',
            'baseline_contract':'Finite RMS/SiLU operators and n0/s1 factors use actual B2 endpoints. B1 clean factor transfer is explicit.',
            'output_projection_scope':'mnorm is the actual saved projection coefficient. Its projection residual remains in the enclosing layer ledger, outside this norm-gate decomposition.',
            'product_baseline_limitation':None if b1_eos is not None else 'product_B2_baseline includes any B2-to-B1 baseline transfer; actual B1 EOS not supplied',
            'prediction':float(prediction.sum()),'actual_native_fused_effect':float(actual.sum()),
            'measured_prediction_minus_actual':float(measured.sum()),
            'terms':{name:float(value.sum()) for name,value in per_token.items()},
            'product_subdivision':{name:float(value.sum()) for name,value in subdivisions.items()},
            'operator_subdivision':{name:float(value.sum()) for name,value in operator_subdivisions.items()},
            'per_token_terms':{name:value.tolist() for name,value in per_token.items()},
            'per_token_measured_prediction_minus_actual':measured.tolist(),
            'per_token_product_subdivision':{name:value.tolist() for name,value in subdivisions.items()},
            'per_token_operator_subdivision':{name:value.tolist() for name,value in operator_subdivisions.items()},
            'per_head':per_head,'endpoint_discrepancies':metrics,
            'geometry_interpretation':'Radius and direction summaries are associations with the observed RMS residual, not proof of an incorrect method or a guaranteed repair.',
            'telescoping_error':float(closure.sum()),'max_abs_token_telescoping_error':float(closure.abs().max()),
            'CPU_algebra_seconds':time.perf_counter()-started,'new_model_calls':0,'new_native_norm_calls':0}
