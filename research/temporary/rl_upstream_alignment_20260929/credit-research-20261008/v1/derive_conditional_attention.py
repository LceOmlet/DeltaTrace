"""Conditional attention identities and owner-interface/cost audit, on CPU.

This derives a possible internal finite rule. It does not implement it, choose
a component from one token, change Q/V/A, or score a candidate's quality.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def main():
    started = time.perf_counter()
    owners = json.loads((HERE/'attention-owner-sources.json').read_bytes())
    config = json.loads((HERE/'gdn-owner-readonly.json').read_bytes())['model_config']['text_config']
    # p is the factual probability of one key. u is exp(s_CF-s_F).
    # a,b,c are arbitrary upstream dot products with O_F,V_F,V_CF.
    p,u = sp.symbols('p u', positive=True)
    a,b,c = sp.symbols('a b c')
    den = 1-p+p*u
    cf = (a-p*b+p*u*c)/den
    exact = a-cf
    coupled = p/den*((b-c)+(1-u)*(c-a))
    content1 = p*(b-c)+p*(1-u)/den*(c-a+p*(b-c))
    checks = {}
    for name,expression in [('conditional_KV_output',exact-coupled),
                            ('content1_with_conditional_softmax',exact-content1),
                            ('conditional_probability_normalization',(1-p)/den+p*u/den-1)]:
        remainder = sp.cancel(expression)
        assert remainder==0, (name,remainder)
        checks[name] = str(remainder)
    # For arbitrary support, normalization gives sum(lambda*delta_s)=tau*delta_logZ.
    tau,z,total,weighted,weighted_v = sp.symbols('tau delta_logZ sum_lambda_delta_s sum_lambda_V_delta_s sum_lambda_V')
    centered = weighted-weighted_v/tau*total
    probability_difference = weighted-z*weighted_v
    remainder = sp.simplify((centered-probability_difference).subs(total,tau*z))
    assert remainder==0
    checks['conditional_query_centering_for_arbitrary_support'] = str(remainder)
    # Local leave-one-row effects need not sum to the all-rows endpoint effect.
    # This symbolic arbitrary bilinear interaction is a protocol check, not a
    # benchmark fixture or evidence about the measured model's quality.
    x0,x1,y0,y1,w = sp.symbols('x0 x1 y0 y1 w')
    loo_sum = w*(x1-x0)*y1+w*x1*(y1-y0)
    joint = w*x1*y1-w*x0*y0
    assert sp.simplify(loo_sum-joint-w*(x1-x0)*(y1-y0))==0
    owner_bindings = {}
    for task,entry in owners['tasks'].items():
        decoder = next(f for f in entry['files'] if f['module']=='qwen35_decoder_finite')
        tree = ast.parse(decoder['text'])
        pullback = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='attention_finite_pullback')
        gate = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_attention_gate_rule')
        owner_bindings[task] = dict(decoder_path=decoder['path'],decoder_sha256=decoder['sha256'],
            recorded_decoder_match=entry['recorded_decoder_match'],
            pullback_signature=ast.unparse(pullback.args),gate_rule=ast.unparse(gate),
            finite_library=entry['finite_library'])
    public = owners['native_FA_public_interface']
    public_tree = ast.parse(public['text'])
    public_contracts = {}
    for class_name, function_name in [('FlashAttnFunc','flash_attn_func'),
                                      ('FlashAttnVarlenFunc','flash_attn_varlen_func')]:
        function = next(n for n in public_tree.body if isinstance(n,ast.FunctionDef)
                        and n.name==function_name)
        owner_class = next(n for n in public_tree.body if isinstance(n,ast.ClassDef)
                           and n.name==class_name)
        forward = next(n for n in owner_class.body if isinstance(n,ast.FunctionDef)
                       and n.name=='forward')
        documentation = ast.get_docstring(function)
        assert 'bottom right corner' in documentation
        dispatch = next(n for n in ast.walk(forward) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Name)
                        and n.func.id in ('_flash_attn_forward','_flash_attn_varlen_forward'))
        request = next(k.value for k in dispatch.keywords if k.arg=='return_softmax')
        assert ast.unparse(request)=='return_softmax and dropout_p > 0'
        public_contracts[function_name] = dict(signature=ast.unparse(function.args),
            documentation=documentation,forward_source=ast.unparse(forward),
            lower_level_return_softmax=ast.unparse(request))
    batch,length,heads,dim,bytes_per = 4,32768,config['num_attention_heads'],config['head_dim'],2
    output_bytes = batch*heads*length*dim*bytes_per
    scalar_row_bytes = batch*heads*length*4
    result = dict(scope=__doc__, checks=checks, owners=owner_bindings,
        evidence={p.name:ref(p) for p in (HERE/'attention-owner-sources.json',
            HERE/'decoder-group-analysis.json',HERE/'credit-probability-bounds.json')},
        conditional_KV=dict(
            definition='Other query/key/value rows are factual; only key/value row i is at its captured reference.',
            u='exp(s_CF_i-s_F_i)=exp(-delta_s_i)',
            denominator='D_i=1-p_i+p_i*u_i',
            output='O_CF=(O_F-p_i*V_F_i+p_i*u_i*V_CF_i)/D_i',
            ordered_rule='Delta O=p_i*Delta V_i + [p_i*(1-u_i)/D_i]*(V_CF_i-O_F+p_i*Delta V_i)',
            coefficients='Value coefficient retains factual p_i. Key coefficient uses exprel(-delta_s_i), factual Q and the conditional normalization above; it is a finite rule, not the factual derivative or a change of joint P0/P1 split.'),
        conditional_query=dict(
            definition='At query row i use factual keys/values except its own reference K_i,V_i. Both Q endpoints use that same conditional key/value background.',
            lambda_definition='lambda_j=(p1_j-p0_j)/(log p1_j-log p0_j), continuous at equal endpoints',
            coefficients='C=sum_j(lambda_j*V_cond_j)/sum_j lambda_j; m_s_j=lambda_j*upstream dot(V_cond_j-C); m_Q=scale*sum_j m_s_j*K_cond_j',
            self_interaction='Do not use independently factual Q and K references at the diagonal: the same hidden-row deletion changes both.',
            gate='The original content1 gate decomposition needs this same conditional O_CF_i for its gate branch, rather than the global all-reference O0_i.'),
        numerical_scope=dict(
            concern='Do not evaluate 1-p by subtracting a nearly certain factual key and infer a stable counterfactual from a rounded p=1.',
            stable_partition='For KV replacements a streaming max/argmax and log-partition excluding the largest key can retain the previously tiny other-key mass. Non-maximum keys have p<=1/2; their exclusive partition does not have this dominant-key cancellation.',
            query_reuse='In causal attention, excluding query i\'s own key leaves the strict past. The pinned public FA documentation confirms bottom-right alignment: preserving each original Q extent and shortening its K/V extent by one shifts the last permitted key to the strict past. This is a contract derivation, not a runtime packed/suffix mapping test. Any implementation must use the original unpadding owner and exact cu_seqlens, not a copied parser or mask implementation.',
            current_status='No primitive, library or official tolerance test was changed or executed by this derivation.'),
        pinned_native_FA_contract=dict(path=public['path'],sha256=public['sha256'],
            functions=public_contracts,
            LSE_access='Public return_attn_probs=True exposes the row log-partition. With dropout_p=0 the pinned forward passes return_softmax=False to the lower-level kernel; requesting LSE therefore does not request the quadratic returned probability matrix at this Python dispatch.',
            limits='This source audit does not measure lower-level allocation, kernel numerical error, zero-key-row behavior, packed suffix correctness or wall time. Those are not claimed as tested. Returned attention probabilities are documented as testing-only and potentially rescaled; they are not used here.'),
        interface_gap=dict(
            existing='The measured owner passes paired global endpoints, V0 and global LSE0/LSE1 into its finite FA callback. The gate branch also uses the globally paired attention O0.',
            required='A conditional rule would need factual V1/O1, exact own-row conditional partitions and conditional gate O0. These values belong in the original finite-attention owner; content0 is not that interface.',
            conservation='For interacting rows, sum of exact local leave-one-row effects need not equal the joint endpoint difference: the symbolic bilinear gap is w*(x1-x0)*(y1-y0). Retain that distinction; do not force closure with a credit multiplier or treat a different result as passing the original joint-conservation diagnostic.',
            composition='The local conditional rule\'s composition with current whole-model finite coefficients remains an unimplemented research question. The accepted downstream Q/V/A, self-target, observation mask, whitening and PPO formulas do not change.'),
        cost_geometry=dict(batch=batch,length=length,heads=heads,head_dim=dim,
            one_BF16_attention_output_bytes=output_bytes,
            one_FP32_per_query_scalar_bytes=scalar_row_bytes,
            global_FP32_attention_matrix_bytes=batch*heads*length*length*4,
            possible_native_work='Two strict-past native FA forward calls per full-attention layer, plus conditional finite tiles in the owner. This is operator work, not two extra whole-model passes or per-token DT calls.',
            required_cost_accounting='Derive and count the complete conditional query and KV tile implementation and live tensors before any GPU candidate. Do not promise that two FA forwards are free or convert FLOPs into wall-clock speed.',
            forbidden='No full attention matrix, source-by-time hidden-state bank, additional per-source model forward, sign correction or extreme-source replacement.'),
        decisions=['Do not return to the single-point GDN candidate.',
            'Do not implement a conditional KV-only patch while leaving query/gate/self-interaction on the global background.',
            'No GPU candidate is launched: the complete owner implementation, combination behavior and tile/live-memory accounting are not yet supplied.'],
        source=ref(Path(__file__)), operations=dict(model=0,DT=0,GPU=0,update=0,production_modified=False),
        elapsed_seconds=time.perf_counter()-started,local_rss_bytes=psutil.Process().memory_info().rss,
        local_python=sys.executable,local_python_version=sys.version)
    (HERE/'conditional-attention-algebra.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks=checks,cost_geometry=result['cost_geometry'],
        operations=result['operations'],elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False))


if __name__=='__main__':
    main()
