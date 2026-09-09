"""True sample batching with actual native FA, own-DT endpoint views and seed packing.

Extra EOS tails lie after each fixed generation and receive no target seed.
The default causal native FA includes them; real prefix invariance is audited.
There is no new attention, model, kernel, objective or native backward here.
"""
import hashlib
import torch
from qwen3_batch_capture import capture_checkpoint_batch_raw
from qwen3_batch_finite import propagate_signed_secant
from qwen3_deferred_replay import NativeLayerReplay
from deferred_validation import DeferredValidation
from qwen_signed_secant_paired_public_fa import EndpointReplay


class BatchedReplayViews:
    def __init__(self,model,master,validation):
        self.native=NativeLayerReplay(model,master,validation=validation)
        self.master=master;self.index=None;self.values=None

    def get(self,index,endpoint):
        if index!=self.index:
            raw=self.native[index];views=[{},{}]
            for key,value in raw.items():
                if value is None:
                    assert key=='p';views[0][key]=views[1][key]=None
                else:
                    assert value.shape[0]==2*self.master['sample_batch']
                    for side in range(2):views[side][key]=value[side::2]
            self.values=views;self.index=index
        return self.values[endpoint]

    def clear(self):self.native.clear();self.values=None;self.index=None


def capture_batch(model,cases,eos_token_id):
    batch=len(cases);assert batch>=1
    length=max(len(c['input_ids']) for c in cases);device=model.device
    ids=torch.full((2*batch,length),eos_token_id,device=device,dtype=torch.long)
    samples=[];positions=[];targets=[];prefix_receipts=[]
    for b,c in enumerate(cases):
        original=torch.tensor(c['input_ids'],device=device,dtype=torch.long)
        assert hashlib.sha256(original.cpu().numpy().tobytes()).hexdigest()==c['input_sha256']
        n=len(original);prompt=c['prompt_length'];assert n-prompt==c['target_length']
        assert int(original[-1])==eos_token_id and prompt>=1
        ids[2*b:2*b+2,:n]=original
        eligible=[c['user_positions'][j] for j in c['keep']];ids[2*b,eligible]=eos_token_id
        samples.extend([b]*(n-prompt));positions.extend(range(prompt-1,n-1));targets.extend(original[prompt:].tolist())
        prefix_receipts.append({'case':f"{c['dataset']}_{c['index']}",'input_sha256':c['input_sha256'],'valid_length':n,'padded_length':length,
                                'tail_tokens':length-n,'tail_token_id':eos_token_id,'tail_target_count':0})
    samples=torch.tensor(samples,device=device);positions=torch.tensor(positions,device=device);targets=torch.tensor(targets,device=device)
    root_calls=[]
    def observe(_module,args,kwargs):
        assert torch.equal(kwargs['input_ids'],ids) and torch.equal(kwargs['attention_mask'],torch.ones_like(ids))
        root_calls.append({'input_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),'shape':list(ids.shape),
                           'sample_batch':batch,'endpoint_batch':2*batch,'tail_policy':'causal right EOS tails; no tail seeds'})
    handle=model.register_forward_pre_hook(observe,with_kwargs=True)
    try:master=capture_checkpoint_batch_raw(model,ids,torch.ones_like(ids),samples,positions,targets)
    finally:handle.remove()
    assert len(root_calls)==1 and master['mask'] is None
    master['root_call']=root_calls[0];master['prefix_receipts']=prefix_receipts
    master['valid_lengths']=[len(c['input_ids']) for c in cases]
    return master


def propagate_batch(model,master,finite_attention):
    validation=DeferredValidation();paired=BatchedReplayViews(model,master,validation)
    endpoints=[]
    for e in range(2):
        view={k:master[k] for k in ('cos','sin','mask','length','sample_batch','target_samples','target_positions')}
        for k in ('last','norm_out'):view[k]=master[k][e::2]
        for k in ('logits','target','target_logprobs32','score16','score32_sum64'):view[k]=master[k][e]
        view['layers']=EndpointReplay(paired,e);endpoints.append(view)
    try:
        result=propagate_signed_secant(model,*endpoints,'rescale',pv_rule='content_P1',finite_attention=finite_attention,validation=validation)
        assert paired.native.calls==paired.native.auxiliary_attention_calls==len(model.model.layers)==36
        result.update(sample_batch=master['sample_batch'],endpoint_batch=2*master['sample_batch'],
                      actual_root=master['root_call'],prefix_receipts=master['prefix_receipts'],
                      checkpoint_retained_tensor_bytes=master['tensor_bytes'],
                      native_layer_replay_calls=paired.native.calls,extra_native_fa_attention_calls=paired.native.auxiliary_attention_calls,
                      native_layer_boundary_checks=paired.native.boundary_checks,public_FA_capture_checks=paired.native.public_capture_checks,
                      target_scores_by_sample=master['scores_by_sample'],
                      signed_sum_scope='complete sample batch; per-sample vectors are separate')
        result=validation.finish(result)
        signed=torch.tensor(result.pop('signed_full_sequence'),dtype=torch.float64)
        assert signed.shape==(master['sample_batch'],master['length']) and torch.isfinite(signed).all()
        for b,n in enumerate(master['valid_lengths']):assert bool(signed[b,n:].eq(0).all())
        return [signed[b,:n] for b,n in enumerate(master['valid_lengths'])],result
    finally:paired.clear()


def attribute_batch(model,cases,eos_token_id,finite_attention):
    master=capture_batch(model,cases,eos_token_id)
    root_peak=torch.cuda.max_memory_allocated()
    vectors,detail=propagate_batch(model,master,finite_attention)
    detail['root_peak_allocated']=root_peak
    return vectors,detail
