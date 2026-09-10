"""Separately charged diagnostics on actual inputs to each compiled boundary.

Always return the candidate output; compare against the previous compiled rule
with its original materialized FP32 inputs. This never selects a layer's rule.
"""
import torch
import qwen3_cast_boundaries as candidate
from compiled_finite_rules import rmsnorm_secant_pullback as reference_norm
from compiled_secant_boundaries import norm_residual_two as reference_two,norm_residual_three as reference_three,attention_layout as reference_layout

enabled=False
records=[]


def compare(name,actual,expected):
    actual=(actual,) if isinstance(actual,torch.Tensor) else actual
    expected=(expected,) if isinstance(expected,torch.Tensor) else expected
    for index,(a,b) in enumerate(zip(actual,expected)):
        records.append({'name':name,'output_index':index,'shape':list(a.shape),'exact':bool(torch.equal(a,b)),
            'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))})


def norm(x0,x1,w,m,eps):
    result=candidate.norm(x0,x1,w,m,eps)
    if enabled:compare('norm',result,reference_norm(x0.float(),x1.float(),w.float(),m,eps))
    return result


def norm_residual_two(x0,x1,w,a,b,r,eps):
    result=candidate.norm_residual_two(x0,x1,w,a,b,r,eps)
    if enabled:compare('two',result,reference_two(x0.float(),x1.float(),w.float(),a,b,r,eps))
    return result


def norm_residual_three(x0,x1,w,a,b,c,r,eps):
    result=candidate.norm_residual_three(x0,x1,w,a,b,c,r,eps)
    if enabled:compare('three',result,reference_three(x0.float(),x1.float(),w.float(),a,b,c,r,eps))
    return result


def qk_norms(q0,q1,qw,mq,qe,k0,k1,kw,mk,ke):
    result=candidate.qk_norms(q0,q1,qw,mq,qe,k0,k1,kw,mk,ke)
    if enabled:compare('qk',result,(reference_norm(q0.float(),q1.float(),qw.float(),mq,qe),reference_norm(k0.float(),k1.float(),kw.float(),mk,ke)))
    return result


def attention_layout(q,k,v,*rest):
    result=candidate.attention_layout(q,k,v,*rest)
    if enabled:compare('layout',result,reference_layout(q.float(),k.float(),v.float(),*rest))
    return result
