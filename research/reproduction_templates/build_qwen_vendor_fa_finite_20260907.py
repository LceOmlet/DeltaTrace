"""Replace only the attribution attention block; preserve actual model replay."""
import ast,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent
base=(A/'qwen_signed_secant_pv_rules.py').read_text()
text=base.replace('midpoint, scaled_probability, attention_layout,','attention_layout,')
text=text.replace('from compiled_native_probability_audit import compiled_pv_audit\n','')
text=text.replace('from compiled_finite_rules import softmax_secant_pullback, rmsnorm_secant_pullback','from compiled_finite_rules import rmsnorm_secant_pullback')
text=text.replace("pv_rule='symmetric'):","pv_rule='content_P1', finite_attention=None, finite_activity=None):",1)
text=text.replace("    assert pv_rule in ['symmetric', 'content_P1', 'content_P0']","    assert pv_rule == 'content_P1' and callable(finite_attention)\n    if finite_activity is None: finite_activity=[]",1)
begin=text.index('        lse = values[\'fa_lse\'].float()');end=text.index('    with torch.no_grad():',begin)
helper='''        lse = values['fa_lse']
        q, k, v = qs.transpose(1,2), ks.transpose(1,2), vs.transpose(1,2)
        assert q.dtype==k.dtype==v.dtype==torch.float16 and h==128
        assert lse.shape==q.shape[:3] and lse.dtype==torch.float32
        assert torch.isfinite(lse).all()
        kr=k.repeat_interleave(groups,dim=1);vr=v.repeat_interleave(groups,dim=1)
        fa_operand_audits.append({'layer':layer.self_attn.layer_idx,
            'endpoint':'before' if len(fa_operand_audits)%2==0 else 'after',
            'actual_native_qkv_and_output_exact':True,'model_dropout':0.0,
            'auxiliary_output_used_by_model':False,'extra_native_attention_calls':0,
            'probability_source':'No global probability matrix. Finite FA extension reconstructs native-LSE probabilities only within tiles.',
            'dense_probability_audit_performed':False,
            'audit_scope':'Exact original RoPE/QKV/output provenance plus actual native public LSE; dense numerical audit is a separate charged development control.'})
        return q,k,v,kr,vr,lse
'''
text=text[:begin]+helper+text[end:]
begin=text.index('            q0, k0, v0, kr0');end=text.index('            mqp = norm_back',begin)
block='''            q0,k0,v0,kr0,vr0,lse0=native_attention_operands(raw0,layer)
            q1,k1,v1,kr1,vr1,lse1=native_attention_operands(raw1,layer)
            n=before['length'];heads=q0.shape[1];h=layer.self_attn.head_dim
            groups=layer.self_attn.num_key_value_groups
            mout=m_concat.view(1,n,heads,h).transpose(1,2)
            activity={'layer':li};finite_activity.append(activity)
            with torch.profiler.record_function('ATTR_VENDOR_FA_FINITE_P1'):
                outputs=finite_attention({'q0':q0,'k0':kr0,'q1':q1,'k1':kr1,
                    'v0':vr0,'u':mout,'lse0':lse0,'lse1':lse1},layer.self_attn.scaling,activity)
            assert all(torch.isfinite(value).all() for value in outputs.values())
            assert (outputs['tau']>0).all()
            # Kernel Q/K outputs already include the attention scale. Existing
            # compiled layout applies unit scale, and sums GQA in FP32.
            q_product=outputs['dq'].float();k_product=outputs['dk'].float();mvr=outputs['dv'].float()
            mqn,mkn,mv=attention_layout(q_product,k_product,mvr,before['cos'],before['sin'],
                groups,1.0,k0.shape[1],v0.shape[1],n,h)
'''
text=text[:begin]+block+text[end:]
text=text.replace("'native_LSE_probability_PV_approximation_explicit': True", "'native_LSE_probability_reconstructed_only_in_finite_tiles': True")
old="            del a, b, q0, k0, v0, kr0, vr0, p0, score0, z0, q1, k1, v1, kr1, vr1, p1, score1, z1, mp, mvr, mz, q_product, k_product, mv"
assert text.count(old)==1
text=text.replace(old,"            del a,b,q0,k0,v0,kr0,vr0,lse0,q1,k1,v1,kr1,vr1,lse1,mvr,q_product,k_product,mv,outputs")
text=text.replace("'variant': variant, 'pv_rule': pv_rule,", "'variant': variant, 'pv_rule': pv_rule, 'finite_attention_activity':finite_activity,",1)
text=text.replace('Same original native endpoint/replay and rescale finite attribution. Per-operator diagnostic-only FP64 credits and unused FP32 diagnostic copies omitted explicitly. Native model, FA and attribution multipliers unchanged; no surrogate gradient.',
 'Original native endpoint/replay preserved. Content-P1 finite attention uses traceable FA-framework extension with FP16 operands/outputs and FP32 accumulators. Remaining finite propagation unchanged. No dense probability/score/finite audit matrices in production. No model or native-autograd substitution.')
text=text.replace('Only PV interaction allocation differs: symmetric, deltaP*V0+P1*deltaV, or deltaP*V1+P0*deltaV. Mixed endpoints are algebraic attribution terms, never claimed as actual model counterfactual forwards. NativeFA probability approximation/global residual remain explicit.',
 'Content P1 only: deltaP*V0+P1*deltaV; finite logmean softmax and symmetric QK allocation. Half precision and actual public native LSE are explicit numerical choices. Mixed endpoints are algebraic terms, never claimed model counterfactual forwards.')
tree=ast.parse(text)
# Actual finite function has no @, bmm, or explicit probability helper; model
# replay and projection GEMMs remain in their original independently named code.
assert not any(isinstance(n,ast.MatMult) for n in ast.walk(tree))
assert 'scaled_probability(' not in text and 'softmax_secant_pullback(' not in text
(A/'qwen_signed_secant_vendor_fa.py').write_text(text,encoding='utf-8')
paired='''"""Original public FA paired checkpoint/replay plus separately named finite op."""
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews,EndpointReplay
from qwen_signed_secant_vendor_fa import propagate_signed_secant

def propagate_paired_secant(model,before,after,progress=None,pv_rule='content_P1',activity=None,finite_attention=None,finite_activity=None):
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    assert before['endpoint_index']==0 and after['endpoint_index']==1
    master=before['paired_checkpoint'];paired=PairedReplayViews(model,master,activity=activity)
    left=dict(before);right=dict(after)
    left['layers']=EndpointReplay(paired,0);right['layers']=EndpointReplay(paired,1)
    try:
        result=propagate_signed_secant(model,left,right,'rescale',progress=progress,pv_rule=pv_rule,
            finite_attention=finite_attention,finite_activity=finite_activity)
        assert paired.native.calls==len(model.model.layers)==36
        assert paired.native.auxiliary_attention_calls==paired.native.calls
        result['extra_native_fa_attention_calls']=paired.native.auxiliary_attention_calls
        result['extra_native_fa_attention_endpoint_trajectories']=2*paired.native.auxiliary_attention_calls
        result['public_FA_capture_checks']=paired.native.public_capture_checks
        for audit in result['native_fa_operand_audits']:
            audit['extra_native_attention_calls']=1
            audit['extra_call_count_scope']='One shared B2 public call per layer; both endpoint audit rows refer to this call.'
        result['native_layer_replay_calls']=paired.native.calls
        result['native_layer_replay_endpoint_trajectories']=2*paired.native.calls
        result['native_transformer_body_replay_equivalents']=2
        result['native_layer_boundary_checks']={'paired_batch':paired.native.boundary_checks}
        result['native_paired_replay_layouts']=paired.layout_records
        result['checkpoint_retained_tensor_bytes']=master['tensor_bytes']
        result['recomputation_scope']='One actual default FA B2 root and36 B2 decoder replays, plus36 unchanged public B2 FA metadata calls. Testing matrix never used. Finite attribution is a separate traceable operator, not native backward.'
        return result
    finally:paired.clear()
'''
ast.parse(paired);(A/'qwen_signed_secant_paired_vendor_fa.py').write_text(paired,encoding='utf-8')
print('Prepared separate finite P1 entry; original model, capture, remaining finite rules preserved.')
