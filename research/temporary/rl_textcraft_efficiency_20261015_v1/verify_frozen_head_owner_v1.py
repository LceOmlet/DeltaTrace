"""Finite operator parity/cost check on an actual captured native head chunk.

Imports the pinned owner and the minimally patched owner; no head/PPO formula
is reimplemented. Equality is exact because only an unused matrix product is
removed. This is not a training-quality test or a full-iteration speed claim.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert torch.cuda.device_count()==1
    a=torch.load(args.fixture,map_location='cpu',weights_only=True)
    original,candidate=load('pinned_head_owner',args.baseline),load('patched_head_owner',args.candidate)
    forward_fixture='hidden_states' in a
    h0=a['hidden_states' if forward_fixture else 'hidden']
    weight=a['vocab_weights'].cuda()
    h=h0.cuda();labels=a['input_ids' if forward_fixture else 'labels'].cuda()
    temperature=a.get('temperature',1.0)
    # When the actual backward fixture exists, retain its native upstream too.
    # The forward fixture permits an earlier operator-only check with explicit
    # fixed nonzero cotangents; it makes no claim about actual PG attribution.
    if not forward_fixture:
        dl=None if a['dlog_probs'] is None else a['dlog_probs'].cuda()
        de=None if a['dentropy'] is None else a['dentropy'].cuda()
    else:
        generator=torch.Generator(device='cpu').manual_seed(20261015)
        dl=(torch.randn(len(h),generator=generator,dtype=torch.float32)*1e-5).cuda()
        de=torch.full((len(h),),-1e-7,device='cuda',dtype=torch.float32)
    kwargs=dict(dlog_probs=dl,dentropy=de,hidden_states=h,vocab_weights=weight,input_ids=labels,temperature=temperature)
    state=dict(scope=__doc__,started_unix=time.time(),fixture=str(args.fixture),
        fixture_kind='actual_native_forward_hidden_with_fixed_operator_cotangents' if forward_fixture else 'actual_native_backward_chunk',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.baseline,args.candidate)},
        hidden_shape=list(h.shape),weight_shape=list(weight.shape),dtype=str(h.dtype),temperature=temperature,
        acceptance='torch.equal for every needed output; no nonzero tolerance',phase='warm_and_check')
    def save():args.output.write_text(json.dumps(state,indent=2)+'\n')
    save()
    torch.cuda.synchronize();start=time.perf_counter()
    g0,v0=original._fused_linear_for_ppo_bwd(**kwargs);torch.cuda.synchronize()
    state['baseline_first_call_seconds']=time.perf_counter()-start
    del v0
    torch.cuda.synchronize();start=time.perf_counter()
    g1,v1=candidate._fused_linear_for_ppo_bwd(**kwargs,compute_vocab_grad=False);torch.cuda.synchronize()
    state['candidate_first_call_seconds']=time.perf_counter()-start
    state['hidden_gradient_equal']=torch.equal(g0,g1)
    state['hidden_gradient_max_abs_diff']=float((g0.float()-g1.float()).abs().max())
    state['discarded_vocab_gradient_is_none']=v1 is None
    if not forward_fixture:
        state['original_matches_saved_native_hidden_gradient']=torch.equal(g0.cpu(),a['native_hidden_gradient'])
    assert state['hidden_gradient_equal'] and v1 is None
    del g0,g1,v1
    state['public_autograd_checks']=[]
    # Includes one full chunk and a tail, the existing 2D/3D owner paths, and
    # the unchanged trainable-head branch. No optimizer/model is constructed.
    for shape,trainable_head in [('2D512',False),('3D515',False),('2D17',True)]:
        if shape=='3D515':
            hh=torch.cat([h,h[:3]],dim=0).unsqueeze(0)
            yy=torch.cat([labels,labels[:3]]).unsqueeze(0)
            dd=None if dl is None else torch.cat([dl,dl[:3]]).unsqueeze(0)
            ee=None if de is None else torch.cat([de,de[:3]]).unsqueeze(0)
        else:
            n=17 if trainable_head else len(h)
            hh,yy=h[:n],labels[:n];dd=None if dl is None else dl[:n];ee=None if de is None else de[:n]
        results=[]
        for module in (original,candidate):
            x=hh.detach().requires_grad_();w=weight.detach().requires_grad_(trainable_head)
            lp,en=module.FusedLinearForPPO()(x,w,yy,temperature)
            outs=[];ups=[]
            if dd is not None:outs.append(lp);ups.append(dd)
            if ee is not None:outs.append(en);ups.append(ee)
            grads=torch.autograd.grad(outs,[x,w] if trainable_head else [x],ups)
            results.append([lp.detach().cpu(),en.detach().cpu()]+[g.detach().cpu() for g in grads])
            del lp,en,grads,x,w
        checks=[torch.equal(x,y) for x,y in zip(*results)]
        state['public_autograd_checks'].append(dict(shape=shape,trainable_head=trainable_head,
            returned_logp_entropy_and_requested_gradients_equal=checks))
        save();assert all(checks)
        del results
    state['phase']='hot_cost_measurement';save()
    # Warm each compiled helper, then interleave order to avoid conflating a
    # one-time compile, clock ramp or a monotonic host load change with speed.
    for _ in range(2):
        z=original._fused_linear_for_ppo_bwd(**kwargs);del z
        z=candidate._fused_linear_for_ppo_bwd(**kwargs,compute_vocab_grad=False);del z
    torch.cuda.synchronize()
    rows=[]
    for block in range(8):
        order=('baseline','candidate') if block%2==0 else ('candidate','baseline')
        for label in order:
            fn=original._fused_linear_for_ppo_bwd if label=='baseline' else candidate._fused_linear_for_ppo_bwd
            extra={} if label=='baseline' else dict(compute_vocab_grad=False)
            torch.cuda.synchronize();tick=time.perf_counter()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                z=fn(**kwargs,**extra);del z
            end.record();torch.cuda.synchronize()
            rows.append(dict(block=block,label=label,calls=10,wall_seconds=time.perf_counter()-tick,
                stream_seconds=begin.elapsed_time(end)/1000))
    med={label:statistics.median(r['wall_seconds']/r['calls'] for r in rows if r['label']==label)
         for label in ('baseline','candidate')}
    state.update(phase='completed',completed_unix=time.time(),cost_blocks=rows,
        median_seconds_per_chunk=med,chunk_backward_speed_ratio=med['baseline']/med['candidate'],
        model_calls=0,DT_calls=0,optimizer_calls=0,training_mutations=0,
        limits=['One head operator cost is not full PPO update or training speed.',
                'Exact comparisons cover the recorded fixtures and owner branches; no general BF16 error bound is inferred.'])
    save();print(json.dumps(state))


if __name__=='__main__':main()
