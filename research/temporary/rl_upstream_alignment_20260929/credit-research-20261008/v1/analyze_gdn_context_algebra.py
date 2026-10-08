"""Owner-bound algebra/cost audit; not a model, credit implementation or quality test."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
import sympy as sp

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
SOURCES = AUDIT/'direct-target-existing-pv-rule-20261008/v1/joint-finite-sources.json'
GDN_CAPTURE = AUDIT/'direct-target-existing-pv-rule-20261008/v1/gdn-v2-results/results/rank0.json'


def fingerprint(path):
    data = path.read_bytes()
    return {'path':str(path), 'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}


def prove_identities(width, chunk):
    """Noncommuting symbolic matrices; no sampled numbers or tensor fixture."""
    def nc(name):
        return sp.Symbol(name, commutative=False)
    checks = []
    def check(name, expression):
        residue = sp.expand(expression)
        if residue != 0:
            raise AssertionError((name, str(residue)))
        checks.append({'name':name, 'expanded_residual':'0'})

    mf,mc,hf,hc,bf,bc = [nc(x) for x in ('Mf','Mc','Hf_prev','Hc_prev','Bf','Bc')]
    dh,dm,db = hf-hc,mf-mc,bf-bc
    check('Coupled finite state difference with factual previous-state form',
          mf*hf+bf-mc*hc-bc-(mf*dh+dm*hf-dm*dh+db))
    check('Same identity with counterfactual previous-state form',
          mf*hf+bf-mc*hc-bc-(mf*dh+dm*hc+db))
    qf,qc,hf,hc = [nc(x) for x in ('qf','qc','Hf','Hc')]
    check('Finite readout including the query/state interaction',
          qf*hf-qc*hc-(qf*(hf-hc)+(qf-qc)*hf-(qf-qc)*(hf-hc)))
    check('Symmetric finite readout, no scalar rescaling',
          qf*hf-qc*hc-((qf+qc)*(hf-hc)+(qf-qc)*(hf+hc))/2)

    a,alpha,beta=sp.symbols('a alpha beta',commutative=True)
    k,kt,u,vt,l,rt,v=[nc(x) for x in ('k','kt','U','Vt','L','Rt','vt')]
    check('Transition product low-rank induction: one new column per position',
          alpha*(1-beta*k*kt)*(a+u*vt)
          -(alpha*a+alpha*u*vt-alpha*beta*k*(a*kt+kt*u*vt)))
    check('Additive state low-rank induction: one new column per position',
          alpha*(1-beta*k*kt)*l*rt+beta*k*v
          -((alpha*l-alpha*beta*k*kt*l)*rt+beta*k*v))

    a0,a1,b0,b1=sp.symbols('a0 a1 b0 b1',commutative=True)
    k0,k1,t0,t1,v0,v1,p0,p1=[nc(x) for x in
        ('k0','k1','k0T','k1T','v0T','v1T','H0prev','H1prev')]
    f=a1*p1-a1*b1*k1*t1*p1+b1*k1*v1
    c=a0*p0-a0*b0*k0*t0*p0+b0*k0*v0
    u0=b0*(v0-a0*t0*p0)
    rule=(a1*(1-b1*k1*t1)*(p1-p0)
          +(a1-a0)*(p0-b1*k1*t0*p0)
          +(k1-k0)*u0-a1*b1*k1*(t1-t0)*p0
          +b1*k1*(v1-v0)+(b1-b0)*k1*(v0-a0*t0*p0))
    check('Complete finite q/k/v/alpha/beta memory rule before cotangent contraction',f-c-rule)

    initial=nc('Hstart')
    mf=[nc('Mf'+str(i)) for i in range(width)]
    mc=[nc('Mc'+str(i)) for i in range(width)]
    bf=[nc('Bf'+str(i)) for i in range(width)]
    bc=[nc('Bc'+str(i)) for i in range(width)]
    f,c=[initial],[initial]
    for i in range(width):
        f.append(mf[i]*f[-1]+bf[i])
        c.append(mc[i]*c[-1]+bc[i])
    total=0
    for i in range(width):
        tail=sp.S.One
        for j in reversed(range(i+1,width)):
            tail=tail*mf[j]
        total += tail*((mf[i]-mc[i])*c[i]+bf[i]-bc[i])
    check('Exact coupled '+str(width)+'-position window; factual future and conditional past',
          f[-1]-c[-1]-total)
    naive=0
    for i in range(width):
        tail=sp.S.One
        for j in reversed(range(i+1,width)):
            tail=tail*mf[j]
        naive += tail*((mf[i]-mc[i])*f[i]+bf[i]-bc[i])
    omitted=sp.expand(f[-1]-c[-1]-naive)
    if omitted == 0:
        raise AssertionError('The independent-position shortcut unexpectedly lost its mixed terms')

    alpha=sp.symbols('a0:'+str(chunk), commutative=True)
    update=[nc('U'+str(i)) for i in range(chunk)]
    value=initial
    for i in range(chunk):
        value=alpha[i]*value+update[i]
    compact=sp.prod(alpha)*initial+sum(sp.prod(alpha[j+1:])*update[j] for j in range(chunk))
    check('Native decay/rank-update state expansion for '+str(chunk)+'-position chunk',value-compact)
    terminal=nc('Dend')
    event=[nc('E'+str(i)) for i in range(chunk)]
    value=terminal
    for i in reversed(range(chunk)):
        value=alpha[i]*(value+event[i])
    compact=sp.prod(alpha)*terminal+sum(sp.prod(alpha[:j+1])*event[j] for j in range(chunk))
    check('Factual reverse state expansion for '+str(chunk)+'-position chunk',value-compact)
    return {
        'scope':'Exact symbolic identities for arbitrary noncommuting operators; not empirical DT quality, an official numerical tolerance, or an implemented token-credit algorithm',
        'sympy_version':sp.__version__, 'checks':checks,
        'independent_position_shortcut':{
            'algebraically_exact':False,
            'omitted_noncommuting_monomials':len(sp.Add.make_args(omitted)),
            'reason':'The same hidden-row intervention changes several adjacent Q/K/V operands; their conditional state histories are coupled.'},
    }


def main():
    started=time.perf_counter()
    owner=json.loads((HERE/'gdn-owner-readonly.json').read_text(encoding='utf-8'))
    source=json.loads(SOURCES.read_text(encoding='utf-8'))
    gdn=json.loads(GDN_CAPTURE.read_text(encoding='utf-8'))['gdn_subops']['30']
    finite=next(x for x in source['files'] if x['module']=='finite_fla_gpu')
    if hashlib.sha256(finite['text'].encode()).hexdigest()!=finite['sha256']:
        raise AssertionError('Captured finite source identity changed')
    config=owner['model_config']['text_config']
    width=config['linear_conv_kernel_dim']
    # The current source explicitly owns 64-token native chunks and 128-wide heads.
    tree=ast.parse(finite['text'])
    mixed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mixed_coefficients')
    def count_calls(nodes):
        return sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mm'
                   for root in nodes for n in ast.walk(root))
    reuse=next(n for n in mixed.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name)
               and n.test.id=='reuse_scalar_products')
    default_mm=count_calls(mixed.body)-count_calls(reuse.body)
    alternative_mm=count_calls(mixed.body)-count_calls(reuse.orelse)
    batch,length,chunk=4,32768,64
    heads=config['linear_num_value_heads']
    head_group=gdn['effective_options']['fla_head_batch_size']
    key,value=config['linear_key_head_dim'],config['linear_value_head_dim']
    chunks=(length+chunk-1)//chunk
    def size(n,bytes_per_element):
        b=n*bytes_per_element
        return {'bytes':b,'GiB':b/(1024**3)}
    state_elements=batch*length*heads*key*value
    chunk_elements=batch*chunks*heads*key*value
    factor_elements=batch*length*heads*key*width
    result={
        'status':'Mathematical/source/cost diagnostic only; no candidate implemented or deployed',
        'unix':time.time(),
        'evidence':{p.name:fingerprint(p) for p in [HERE/'gdn-owner-readonly.json',SOURCES,GDN_CAPTURE,HERE/'manifest.json']},
        'actual_owner_binding':{
            'model_config':owner['model_config']['path'],
            'model_config_sha256':owner['model_config']['sha256'],
            'modeling_sha256':owner['owners']['transformers/models/qwen3_5/modeling_qwen3_5.py']['sha256'],
            'fla_chunk_sha256':owner['owners']['fla/ops/gated_delta_rule/chunk.py']['sha256'],
            'fla_state_sha256':owner['owners']['fla/ops/common/chunk_delta_h.py']['sha256'],
            'finite_fla_sha256':finite['sha256'],
            'conv_width':width,'head_key_dim':key,'head_value_dim':value,'value_heads':heads,
        },
        'algebra':prove_identities(width,chunk),
        'root_aligned_possibility':{
            'scope':'Finite GDN memory effect at a given incoming output cotangent. Changing one layer-input row affects a conditional window of width4. This is not an exact original-token/world counterfactual after all model layers.',
            'state':'H_j = alpha_j (I-beta_j k_j k_j^T) H_(j-1) + beta_j k_j v_j^T',
            'coupled_window':'Delta H_end = sum_j (product_(l=end..j+1) M_l^F) [Delta M_j H_(j-1)^conditional + Delta B_j]. The conditional past must be preserved inside the four affected positions.',
            'rank_factorization':'A width-w transition product is a I + U V^T with rank(U V^T)<=w; its additive state update has rank<=w. Induction: multiplying alpha(I-beta k k^T) adds at most one rank-one factor to each part.',
            'consequence':'The conditional window can be represented without one dense KxV counterfactual state per source/time position. This algebra does not yet supply an efficient finite pullback implementation.',
            'compact_forward_query':'H_(t-1)^T a_t = (prod_(l=start..t-1) alpha_l) H_start^T a_t + sum_(j=start..t-1) (prod_(l=j+1..t-1) alpha_l) (k_j^T a_t) u_j. Original captured u_j is v_new, not the uncorrected value v_j.',
            'compact_reverse_query':'Lambda_t^T a_t = (prod_(l=t+1..end) alpha_l) D_end^T a_t + sum_(j=t+1..end) (prod_(l=t+1..j) alpha_l) [scale (q_j^T a_t) do_j - beta_j (k_j^T a_t) L_j], where L_j=(Lambda_j+scale q_j do_j^T)^T k_j.',
            'owner_reuse':'Existing native h/v_new and factual native dh_end/dU_WY (L=A^T dU_WY) provide the chunk data. No copied native forward is needed. The current primitive APIs do not directly return all shifted conditional-window finite coefficients.',
            'conditional_finite_coefficients':{
                'premise':'Within one conditional window, endpoint0 changes only the selected layer-input row and its width4 conv consequences; endpoint1 is factual. All other layer-input rows and the incoming state are factual. A fixed memory-output cotangent do is supplied by the existing DT consumer.',
                'shared_adjoint':'P_j^1=Lambda_j^1+scale*q_j^1*do_j^T, L_j=(P_j^1)^T k_j^1; Lambda_(j-1)^1=(M_j^1)^T P_j^1.',
                'reference_state':'H_j^0=alpha_j^0 H_(j-1)^0+k_j^0 (u_j^0)^T; u_j^0=beta_j^0 (v_j^0-alpha_j^0 (H_(j-1)^0)^T k_j^0). H^0 is the conditional window state, not the global all-EOS state.',
                'q':'m_q_j=scale*H_j^0*do_j',
                'v':'m_v_j=beta_j^1*L_j',
                'k':'m_k_j=P_j^1*u_j^0-alpha_j^1*beta_j^1*H_(j-1)^0*L_j',
                'beta':'m_beta_j=(v_j^0-alpha_j^0*(H_(j-1)^0)^T*k_j^0)^T*L_j',
                'alpha':'m_alpha_j=<P_j^1,H_(j-1)^0>-beta_j^1*L_j^T*(H_(j-1)^0)^T*k_j^0',
                'meaning':'These are obtained by contracting the symbolically checked complete finite state difference. They are not ordinary derivatives and do not require dividing a small hidden change or rescaling output credit. Original exp/log-gate, Q/K normalization and conv/SiLU finite rules remain needed at their owner boundaries.',
                'endpoint_order':'The ordered rule is an algebraic intermediate. Preserve the existing symmetric endpoint convention by also evaluating the reversed conditional window. Both orientations share the factual future adjoint after the last changed operand: all later memory operators and fixed do are the same. The reversed orientation needs only the conditional reverse within width4, not an extra whole-sequence/world backward.',
                'limitation':'Exactness here is the weighted finite memory effect for the specified local conditional window. It does not turn a finite-chain approximation across all hidden/model layers into an exact original-token counterfactual.'},
            'numerics':'Use strict causal/future products in these formulas. Do not obtain an exclusive state by dividing by alpha or exp(g), and do not apply output conservation/sign corrections.',
            'remaining_work':'The complete memory coefficients are derived. Lower their conditional-window states and adjoints to the compact native chunk contractions, account for boundary-crossing width4 windows and the symmetric orientation, and count actual contractions/live tensors before an owner-layer candidate implementation. Original norm/gate and other-layer approximation remain distinct and must be evaluated on the frozen collection.',
        },
        'capacity_geometry':{'batch':batch,'length':length,'heads':heads,'key_dim':key,'value_dim':value,'chunk':chunk,'chunks':chunks},
        'memory_arithmetic':{
            'one_dense_state_per_time_FP16':size(state_elements,2),
            'one_dense_state_per_time_FP32':size(state_elements,4),
            'one_original_chunk_state_FP16':size(chunk_elements,2),
            'one_original_chunk_state_FP32':size(chunk_elements,4),
            'four_materialized_rank_w_factors_FP32_full_heads':size(4*factor_elements,4),
            'four_materialized_rank_w_factors_FP32_head8':size(4*factor_elements*8//heads,4),
            'scope':'Storage arithmetic for one layer; not measured RAM/VRAM or total runtime peak. Even low-rank factors must not all be materialized across time.'},
        'existing_mixed_work':{
            'default_mm_calls_per_orientation':default_mm,
            'reuse_scalar_products_mm_calls_per_orientation':alternative_mm,
            'current_symmetric_orientations':2,
            'current_head_group_size':head_group,
            'current_head_groups':heads//head_group,
            'default_mm_calls_per_symmetric_head_group':2*default_mm,
            'default_mm_calls_all_current_head_groups':2*default_mm*(heads//head_group),
            'shapes_per_orientation':{'C_by_C_times_C_by_K':11,'C_by_K_times_K_by_K':7},
            'analytic_GEMM_FLOP_per_orientation':batch*heads*chunks*(11*2*chunk*chunk*key+7*2*chunk*key*key),
            'scope':'18 vendor GEMMs plus one finite scan and two native adjoint stages per orientation/head group. Source-level call/work count, not a measured kernel trace. FLOPs above cover all32 heads. Excludes norms, projection, transport and compiler. FLOPs are not wall-clock predictions.'},
        'decisions':[
            'No top-negative-source selection or extra whole-model single-deletion refinement.',
            'No replacement of the finite operator by the factual derivative, or V-only substitution.',
            'No independent single-position shortcut across the width4 convolution window.',
            'No dense per-position state bank or eager full-sequence rank-factor bank.',
            'Retain the original frozen development/test corpus and original author quality evaluation; no quality claim from these identities.'
        ],
        'operations':{'new_model':0,'new_DT':0,'new_native_operator':0,'new_backward':0,'new_optimizer':0,
                      'new_GPU_job_started':False,'production_modified':False,'credit_repaired':False},
    }
    if default_mm!=18 or alternative_mm!=15:
        raise AssertionError(('Unexpected owner call count',default_mm,alternative_mm))
    result['cpu_seconds']=time.perf_counter()-started
    memory=psutil.Process().memory_info()
    result['local_resources']={'rss_bytes':memory.rss,
        'peak_working_set_bytes':getattr(memory,'peak_wset',None),
        'torch_imported':'torch' in sys.modules,
        'model_or_FLA_imported':any(n.startswith(('transformers.models.qwen','fla.')) for n in sys.modules)}
    out=HERE/'gdn-context-algebra-cost.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'proofs':len(result['algebra']['checks']),
                      'memory':result['memory_arithmetic'],'existing_mixed_work':result['existing_mixed_work'],
                      'seconds':result['cpu_seconds'],'status':result['status']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
