"""Actual GDN16-term ledger extracted from the previously executed MH19/6 study.

CPU copies, elementwise contractions/reductions only. No model, GEMM, replay,
finite operator or gradient approximation. Signs are prediction minus actual.
Token groups label hidden-coordinate positions, not independent source causes.
"""
import torch

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().to('cpu',copy=True)
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
    return x

def nbytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(nbytes(v) for v in x)
    return 0

def delta(left,right):return {g:{k:left[g][k].double()-right[g][k].double() for k in left[g]} for g in left}

def endpoint(x,i):return {g:{k:v[i::2] for k,v in vs.items()} for g,vs in x.items()}

def dot(m,x,token_axis=1):
    assert m.shape==x.shape,(m.shape,x.shape)
    return float((m.double()*x.double()).sum())


def features(d,c,e):
    out=cpu({'d':{k:d[k] for k in ['input_norm_input','input_norm_output','post_norm_input','post_norm_output','mlp_output','output']},
        'c':{k:c[k] for k in ['input','projected_qkv','conv_output','raw_q','raw_k','a','b','z','norm_output','output']},'e':e})
    out['d'].update({k:cpu(d[k]) for k in ['gate_output','up_output','silu_output','down_input']})
    return out


def token_dot(m,x,token_axis=1):
    assert m.shape==x.shape and m.ndim>=3,(m.shape,x.shape)
    assert m.device.type==x.device.type=='cpu'
    assert token_axis in (1,2)
    return (m.double()*x.double()).sum(tuple(i for i in range(1,m.ndim) if i!=token_axis))

def pack_gdn(upstream,new,terms):
    g=terms['mixer']
    return cpu({'upstream':upstream,'input':new,'m_mlp_norm_output':terms['m_mlp_norm_output'],
        'm_mixer_output':terms['m_mixer_output'],'m_mixer_input':terms['m_mixer_input'],
        **{k:g[k] for k in ['mnorm','mo_before_cast','mo_native','mz','coeff','mq','mk','mb','ma','mconv']},
        'mprojected':g['mprojected'][1::2]})

def _decompose(m,change,dot,tokenwise=False):
    """Existing16term ledger; explicitly convert EVERY term to prediction-actual."""
    d,c,e=change['d'],change['c'],change['e'];f=m['coeff'];u=m['upstream'];mm=m['m_mixer_output'];mi=m['m_mixer_input'];ml=m['m_mlp_norm_output']
    repeat=c['raw_q'].shape[2]//m['mq'].shape[2];assert repeat>=1
    qgroups=c['raw_q'].reshape(*c['raw_q'].shape[:2],m['mq'].shape[2],repeat,c['raw_q'].shape[-1])
    kgroups=c['raw_k'].reshape(*c['raw_k'].shape[:2],m['mk'].shape[2],repeat,c['raw_k'].shape[-1])
    assert bool(qgroups.eq(qgroups[:,:,:,:1,:]).all()) and bool(kgroups.eq(kgroups[:,:,:,:1,:]).all())
    qraw=dot(m['mq'],qgroups[:,:,:,0,:]);kraw=dot(m['mk'],kgroups[:,:,:,0,:]);qnorm=dot(f['q'],e['q']);knorm=dot(f['k'],e['k'])
    v=dot(f['v'],e['v']);beta=dot(f['beta'],e['beta']);g=dot(f['g'],e['raw_g']);mb=dot(m['mb'],c['b']);ma=dot(m['ma'],c['a']);mz=dot(m['mz'],c['z'])
    conv=dot(m['mconv'],c['conv_output'],token_axis=2);projected=dot(m['mprojected'],c['projected_qkv'],token_axis=2);normgate=dot(m['mnorm'],c['norm_output'])
    mo=dot(m['mo_before_cast'],e['o']);monative=dot(m['mo_native'],e['o'])
    old={
        'decoder_output_residual_rounding':dot(u,d['output'])-dot(u,d['post_norm_input'])-dot(u,d['mlp_output']),
        'MLP_combined':dot(u,d['mlp_output'])-dot(ml,d['post_norm_output']),
        'post_RMSNorm':dot(ml,d['post_norm_output'])-dot(mm.double()-u.double(),d['post_norm_input']),
        'mixer_residual_rounding':dot(mm,d['post_norm_input'])-dot(mm,d['input_norm_input'])-dot(mm,c['output']),
        'GDN_output_projection':dot(mm,c['output'])-normgate,'GDN_fused_norm_gate':normgate-mo-mz,
        'GDN_mo_BF16_cast':mo-monative,'GDN_FLA_including_raw_g_exp':monative-qnorm-knorm-v-beta-g,
        'GDN_QK_L2_and_head_fold':qnorm+knorm-qraw-kraw,'GDN_beta_sigmoid':beta-mb,'GDN_raw_g_parameter_map':g-ma,
        'GDN_conv_output_split':qraw+kraw+v-conv,'GDN_conv_linear_and_SiLU':conv-projected,
        'GDN_input_projections':projected+mz+mb+ma-dot(mi,c['input']),'mixer_input_alias':dot(mi,c['input'])-dot(mi,d['input_norm_output']),
        'input_RMSNorm':dot(mi,d['input_norm_output'])-dot(m['input'].double()-mm.double(),d['input_norm_input'])}
    terms={k:-value for k,value in old.items()};actual=dot(u,d['output']);pred=dot(m['input'],d['input_norm_input'])
    result={'sign_convention':'prediction_minus_actual','output_contraction':actual,'input_contraction':pred,
        'prediction_minus_actual':pred-actual,'actual_minus_predicted':actual-pred,'terms':terms,
        'telescoping_error':sum(terms.values())-(pred-actual),'FLA_branch_contractions':{'q':qnorm,'k':knorm,'v':v,'beta':beta,'raw_g':g},
        'norm_gate_branch_contractions':{'normalized_memory_before_cast':mo,'gate':mz}}
    assert (float(result['telescoping_error'].abs().max()) if tokenwise else abs(result['telescoping_error']))<1e-7,result['telescoping_error']
    return result


def decompose(m,change):
    return _decompose(m,change,dot)


def decompose_tokens(m,change):
    result=_decompose(m,change,token_dot,tokenwise=True)
    expected=change['d']['input_norm_input'].shape[:2]
    assert all(v.shape==expected for v in result['terms'].values())
    assert result['input_contraction'].shape==result['output_contraction'].shape==expected
    return result
