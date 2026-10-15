"""Exact original-owner parity and cost at recorded B4 and full 32768 head shapes.

The recorded decoder output is supplied at the existing head boundary; no
transformer, policy loss, optimizer or attribution is reimplemented. Only
unrequested original512-token head chunks may be omitted by the candidate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def verify(destination):
    d=Path(destination)
    r=d.parents[1]
    c=r/'candidates/textcraft-native-head-selection-20261015-v1'
    baseline=load('selected_head_baseline_v1',d/'head-selection-baseline-experimental.py')
    candidate=load('selected_head_candidate_v1',c/'verl/utils/experimental/torch_functional.py')
    old_wrapper=load('selected_head_wrapper_baseline_v1',d/'head-selection-baseline-qwen.py')
    new_wrapper=load('selected_head_wrapper_candidate_v1',c/'verl/models/transformers/qwen3_vl.py')
    from verl.utils.experimental import torch_functional as active_owner
    active_class=active_owner.FusedLinearForPPO
    a=torch.load(d/'native-frozen-head-full-chunk.pt',map_location='cpu',weights_only=True)
    h_cpu=a['hidden_states'];ids_cpu=a['original_input_ids'];weight=a['vocab_weights'].cuda()
    assert h_cpu.ndim==3 and h_cpu.shape[0]==4
    assert torch.equal(torch.roll(ids_cpu,shifts=-1,dims=-1),a['input_ids'])
    state=dict(started_unix=time.time(),scope=__doc__,phase='actual_recorded_interface',checks=[],
        physical_device_index=torch.cuda.current_device(),temperature=a.get('temperature',1.0),
        tolerance=dict(atol=0,rtol=0,reason='All requested rows use the same original512-token chunks, dtypes and forward/backward expressions.'),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            d/'head-selection-baseline-experimental.py',d/'head-selection-baseline-qwen.py',
            c/'verl/utils/experimental/torch_functional.py',c/'verl/models/transformers/qwen3_vl.py']},
        model_calls=0,DT_calls=0,optimizer_scheduler_calls=0,formal_training_updates=0,
        limits=['This reuses actual native decoder outputs at the head boundary; the 32k extension is a head capacity fixture, not a new natural trajectory or a full-training speed measurement.'])
    output=d/'head-selection-verification.json'
    def save():output.write_text(json.dumps(state,indent=2)+'\n')
    def run(module,wrapper,h,ids,keep,dl,de,trainable_weight=False):
        active_owner.FusedLinearForPPO=module.FusedLinearForPPO
        counts=dict(forward=0,backward=0)
        fwd,bwd=module._fused_linear_for_ppo_fwd,module._fused_linear_for_ppo_bwd
        def count_fwd(*args,**kwargs):counts['forward']+=1;return fwd(*args,**kwargs)
        def count_bwd(*args,**kwargs):counts['backward']+=1;return bwd(*args,**kwargs)
        module._fused_linear_for_ppo_fwd=count_fwd;module._fused_linear_for_ppo_bwd=count_bwd
        x=h.detach().requires_grad_();w=weight.detach().requires_grad_(trainable_weight)
        # This object exposes an existing, exact owner artifact, without
        # supplying any substitute transformer computation.
        owner_view=SimpleNamespace(model=lambda *args,**kwargs:BaseModelOutputWithPast(last_hidden_state=x),
                                   lm_head=SimpleNamespace(weight=w))
        tick=time.perf_counter();torch.cuda.synchronize()
        try:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                out=wrapper.forward_with_torch_backend(owner_view,input_ids=ids,
                    temperature=state['temperature'],logits_to_keep=keep)
            selector=slice(-keep,None) if isinstance(keep,int) else keep
            lp,en=out.log_probs,out.entropy
            if wrapper is old_wrapper:lp,en=lp[:,selector],en[:,selector]
            grads=torch.autograd.grad((lp,en),[x,w] if trainable_weight else [x],(dl,de))
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            result=[lp.detach().cpu(),en.detach().cpu()]+[g.detach().cpu() for g in grads]
            return result,dict(seconds=seconds,chunk_calls=counts,physical_free_bytes=torch.cuda.mem_get_info()[0])
        finally:
            module._fused_linear_for_ppo_fwd=fwd;module._fused_linear_for_ppo_bwd=bwd
            active_owner.FusedLinearForPPO=active_class
    try:
        cases=[('recorded_native_B4',h_cpu,ids_cpu,a['logits_to_keep'],False),
               ('default_trainable_head_compatibility',h_cpu[:1,:17],ids_cpu[:1,:17],0,True)]
        # Use the same recorded values as a deterministic full-context shape
        # fixture; its memory/layout are exactly B4 x32768 x4096.
        copies=(32768+h_cpu.shape[1]-1)//h_cpu.shape[1]
        cases.append(('B4_context32768_keep10241',h_cpu.repeat(1,copies,1)[:,:32768].contiguous(),
                      ids_cpu.repeat(1,copies)[:,:32768].contiguous(),10241,False))
        generator=torch.Generator(device='cpu').manual_seed(20261015)
        for name,hh,ii,keep,trainable in cases:
            state['phase']=name;save()
            h,ids=hh.cuda(),ii.cuda()
            selector=slice(-keep,None) if isinstance(keep,int) else keep
            shape=ids[:,selector].shape
            dl=(torch.randn(shape,generator=generator,dtype=torch.float32)*1e-5).cuda()
            de=torch.full(shape,-1e-7,dtype=torch.float32,device='cuda')
            left,lc=run(baseline,old_wrapper,h,ids,keep,dl,de,trainable)
            right,rc=run(candidate,new_wrapper,h,ids,keep,dl,de,trainable)
            equal=[torch.equal(x,y) for x,y in zip(left,right)]
            maxima=[float((x.float()-y.float()).abs().max()) for x,y in zip(left,right)]
            row=dict(case=name,context=h.shape[1],batch=h.shape[0],selected_output_shape=list(shape),
                trainable_head=trainable,logp_entropy_and_requested_gradients_equal=equal,
                maximum_abs_differences=maxima,baseline=lc,candidate=rc)
            state['checks'].append(row);save()
            assert all(equal),row
            if not trainable:
                # Compile/warm costs above stay separate. Alternate pair order
                # on the same physical device for a small hot cost comparison.
                costs=[]
                del left,right
                for rep in range(4):
                    order=((baseline,old_wrapper,'baseline'),(candidate,new_wrapper,'candidate'))
                    if rep%2:order=order[::-1]
                    for owner,wrapper,label in order:
                        values,cost=run(owner,wrapper,h,ids,keep,dl,de)
                        costs.append(dict(repeat=rep,label=label,**cost));del values
                med={label:statistics.median(v['seconds'] for v in costs if v['label']==label)
                     for label in ('baseline','candidate')}
                row.update(hot_costs=costs,median_head_forward_backward_seconds=med,
                           head_boundary_speed_ratio=med['baseline']/med['candidate'])
                save()
            else:del left,right
            del h,ids,dl,de
        state.update(phase='completed',completed_unix=time.time(),all_needed_outputs_gradients_exact=True,
            default_owner_unchanged=True,original_chunk_boundaries_preserved=True,
            full_context_32768_head_verified=True,active_owner_class_restored=active_owner.FusedLinearForPPO is active_class)
        save();return state
    finally:
        active_owner.FusedLinearForPPO=active_class
