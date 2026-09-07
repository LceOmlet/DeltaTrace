"""Generalize P1 leading sample dimension, retaining native FA and finite rules."""
import ast
from pathlib import Path
A=Path(__file__).resolve().parent
old=(A/'compiled_secant_boundaries.py').read_text();tree=ast.parse(old)
fn=lambda name:ast.get_source_segment(old,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
layout=fn('attention_layout_rule').replace('reshape(1,','reshape(q_product.shape[0],')
s='"""Existing finite GQA/RoPE layout, generalized leading sample dimension.\nOfficial torch.compile only; not a model or FA kernel.\n"""\nimport torch\nfrom compiled_secant_boundaries import rotation_transpose\n'+layout+'''
_layout=torch.compile(attention_layout_rule,fullgraph=True,dynamic=True,backend='inductor')
def attention_layout(*args):
    with torch.profiler.record_function('ATTR_COMPILED_ATTENTION_LAYOUT_BATCHED'):
        return _layout(*args)
'''
ast.parse(s);(A/'compiled_secant_batched_layout.py').write_text(s,encoding='utf-8')
s=(A/'qwen_signed_secant_pv_rules.py').read_text()
s=s.replace('midpoint, scaled_probability, attention_layout, norm_residual_two','midpoint, scaled_probability, norm_residual_two',1)
s='from compiled_secant_batched_layout import attention_layout\n'+s
s=s.replace("    device = next(model.parameters()).device", "    batch_size=before['last'].shape[0]\n    assert batch_size==len(before['prompt_len'])==len(before['actual_lengths'])\n    device = next(model.parameters()).device",1)
s=s.replace('.view(1, n,', '.view(batch_size, n,').replace('.reshape(1, n,','.reshape(batch_size, n,')
s=s.replace("        start = before['prompt_len'] - 1\n        m_final = torch.zeros_like(f(before['norm_out']))\n        m_final[0, start:-1] = native_half_linear(seed, model.lm_head.weight)","""        m_final = torch.zeros_like(f(before['norm_out']))
        # Concatenated valid response logits have independent row seeds; no padded loss.
        final_seed=native_half_linear(seed, model.lm_head.weight)
        offset=0
        for sample,(plen,length) in enumerate(zip(before['prompt_len'],before['actual_lengths'])):
            count=length-plen
            m_final[sample,plen-1:length-1]=final_seed[offset:offset+count]
            offset+=count
        assert offset==final_seed.shape[0]
        del final_seed""")
s=s.replace('.sum(-1).flatten()', '.sum(-1)')
s=s.replace('total = float(signed.sum())','total = signed.sum(-1).cpu()')
s=s.replace("'target_delta_score32_sum64': g_delta", "'target_delta_score32_sum64': g_delta.tolist()")
s=s.replace("'target_delta_score16': after['score16'] - before['score16']", "'target_delta_score16': (after['score16'] - before['score16']).tolist()")
s=s.replace("'signed_sum': total, 'unassigned_total': g_delta - total", "'signed_sum': total.tolist(), 'unassigned_total': (g_delta - total).tolist()")
ast.parse(s);(A/'qwen_signed_secant_batched_pv.py').write_text(s,encoding='utf-8')
old=(A/'qwen_signed_secant_native_paired_pv_rules.py').read_text();tree=ast.parse(old)
capture=ast.get_source_segment(old,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_checkpoint_pair_raw'))
capture=capture.replace('def capture_checkpoint_pair_raw(model,ids,mask,prompt_len):','def capture_checkpoint_batch_raw(model,ids,mask,prompt_lens,lengths):')
capture=capture.replace('assert ids.shape[0]==mask.shape[0]==2','assert ids.shape[0]==mask.shape[0]==len(prompt_lens)==len(lengths)')
start=capture.index('            logits=result.logits[:,prompt_len-1:-1]');end=capture.index('        finally:',start)
capture=capture[:start]+'''            for key in ['score16','score32_sum64','target_logprobs32','logits','target']:cache[key]=[]
            for sample,(plen,length) in enumerate(zip(prompt_lens,lengths)):
                assert 0<plen<length<=ids.shape[1]
                logits=result.logits[sample,plen-1:length-1];target=ids[sample,plen:length]
                lp16=logits.log_softmax(-1).gather(1,target[:,None]).flatten()
                lp32=logits.float().log_softmax(-1).gather(1,target[:,None]).flatten()
                cache['score16'].append(float(lp16.sum()))
                cache['score32_sum64'].append(float(lp32.double().sum()))
                for key,value in [('target_logprobs32',lp32),('logits',logits),('target',target)]:cache[key].append(copy(value))
            cache['prompt_len']=list(prompt_lens);cache['actual_lengths']=list(lengths);cache['length']=ids.shape[1]
            del result,logits,target,lp16,lp32
'''+capture[end:]
s='''"""True multi-example P1 with unmodified public FA and explicit causal padding.

Each distinct example has EOS/input endpoints and its own fixed response seed.
Right-side EOS after the complete response is computed but never scored. This
requires the validated purely causal Qwen3 configuration; no left padding,
cross-example attention, prompt truncation or invented response token is used.
"""
import time
from qwen_public_fa_layer_replay import NativeLayerReplay
from qwen_signed_secant_batched_pv import propagate_signed_secant
'''+capture+'''

def capture_batch(model,requests,eos_token_id):
    """requests: before_ids[1,N], after_ids[1,N], prompt_len, one per example."""
    import torch
    assert requests and eos_token_id is not None
    lengths=[a.shape[1] for _,a,_ in requests];plens=[p for _,_,p in requests]
    batch=len(requests);n=max(lengths)
    packed=torch.full((2*batch,n),eos_token_id,device=requests[0][1].device,dtype=torch.long)
    for i,(before,after,plen) in enumerate(requests):
        assert before.shape==after.shape==(1,lengths[i]) and 0<plen<lengths[i]
        assert torch.equal(before[:,plen:],after[:,plen:])
        packed[i,:lengths[i]]=before[0];packed[batch+i,:lengths[i]]=after[0]
    master=capture_checkpoint_batch_raw(model,packed,torch.ones_like(packed),plens*2,lengths*2)
    assert master['mask'] is None and master['cos'].shape[0]==master['sin'].shape[0]==1
    endpoints=[]
    for endpoint in range(2):
        selection=slice(endpoint*batch,(endpoint+1)*batch)
        view={k:master[k] for k in ['cos','sin','mask','length']}
        view.update(prompt_len=plens,actual_lengths=lengths,endpoint_index=endpoint,paired_checkpoint=master)
        for key in ['last','norm_out']:view[key]=master[key][selection]
        for key in ['logits','target','target_logprobs32']:view[key]=torch.cat(master[key][selection],0)
        for key in ['score16','score32_sum64']:view[key]=torch.tensor(master[key][selection],dtype=torch.float64)
        endpoints.append(view)
    return tuple(endpoints)

class BatchReplay:
    def __init__(self,model,master,batch,activity):
        self.native=NativeLayerReplay(model,master,activity=activity);self.batch=batch
    def get(self,index,endpoint):
        raw=self.native[index];part=slice(endpoint*self.batch,(endpoint+1)*self.batch)
        return {key:value[part] if value is not None else None for key,value in raw.items()}
    def clear(self):self.native.clear()

class Endpoint:
    def __init__(self,paired,side):self.paired=paired;self.side=side
    def __getitem__(self,index):return self.paired.get(index,self.side)

def propagate_batch(model,before,after,pv_rule='content_P1',activity=None):
    import torch
    assert before['paired_checkpoint'] is after['paired_checkpoint']
    batch=len(before['prompt_len']);master=before['paired_checkpoint']
    paired=BatchReplay(model,master,batch,activity)
    left=dict(before);right=dict(after);left['layers']=Endpoint(paired,0);right['layers']=Endpoint(paired,1)
    try:
        result=propagate_signed_secant(model,left,right,'rescale',pv_rule=pv_rule)
        assert paired.native.calls==paired.native.auxiliary_attention_calls==len(model.model.layers)
        result.update(example_batch_size=batch,physical_endpoint_batch_size=2*batch,
            actual_lengths=before['actual_lengths'],padded_length=before['length'],prompt_lengths=before['prompt_len'],
            native_layer_replay_calls=paired.native.calls,native_layer_replay_endpoint_trajectories=2*batch*paired.native.calls,
            extra_native_fa_attention_calls=paired.native.auxiliary_attention_calls,
            extra_native_fa_attention_endpoint_trajectories=2*batch*paired.native.auxiliary_attention_calls,
            public_FA_capture_checks=paired.native.public_capture_checks,
            native_layer_boundary_checks=paired.native.boundary_checks,
            endpoint_scores32={'before':before['score32_sum64'].tolist(),'after':after['score32_sum64'].tolist()},
            padding_scope='Only trailing EOS after each complete original response, same at both endpoints. All causal work charged; no padded token loss. Original generation retained per example.',
            partial_recomputation='One actual root call and one actual replay per layer at batch2B; one additional original public FA call per replay. Same P1 arithmetic generalized over independent leading sample dimension; no model/native backward replacement.')
        for sample,length in enumerate(before['actual_lengths']):
            assert all(v==0 for v in result['signed_full_sequence'][sample][length:])
        for audit in result['native_fa_operand_audits']:
            audit.update(extra_native_attention_calls=1,
                probability_source='Explicit finite attribution P from actual QKV and documented auxiliary public FA LSE. Auxiliary output exactly checked, never replaces model output.',
                extra_call_count_scope='One shared batch2B physical auxiliary call per layer; both endpoint rows describe the same call.')
        return result
    finally:paired.clear()
'''
ast.parse(s);(A/'qwen_signed_secant_batched_public_fa.py').write_text(s,encoding='utf-8')
print('Prepared true heterogeneous-example P1 runtime; GPU validation pending.')
