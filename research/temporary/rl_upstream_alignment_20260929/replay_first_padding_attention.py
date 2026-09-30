"""Replay the first differing official Qwen layer on saved valid-token prefixes.

All computations use installed Qwen and PyTorch operators; this is diagnostics,
not a new model path, training configuration, or numerical acceptance threshold.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

OUT=Path(os.environ['PADDING_DIAGNOSTIC_DIR'])
PREVIOUS=OUT.parent/'layer-localization-v2'
LENGTHS=[512,1024,2048,4096]
MODEL=Path(os.environ['MODEL_PATH'])


def sample(value, axis=1):
    if axis==1:
        return torch.stack([value[i,value.shape[1]-n:value.shape[1]-n+512] for i,n in enumerate(LENGTHS)]).detach().cpu().clone()
    return torch.stack([value[i,:,value.shape[2]-n:value.shape[2]-n+512] for i,n in enumerate(LENGTHS)]).detach().cpu().clone()


def difference(x,y):
    d=x.float()-y.float()
    return dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),
                reference_rms=float(y.float().square().mean().sqrt()),exact=bool(torch.equal(x,y)))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True,attn_implementation='sdpa').text_config
    cfg._attn_implementation='sdpa'
    with torch.device('meta'):
        layer=Qwen3_5DecoderLayer(cfg,3)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    prefix='model.language_model.layers.3.'
    state={}
    for key,shard in index.items():
        if key.startswith(prefix):
            with safe_open(MODEL/shard,framework='pt',device='cpu') as reader:
                state[key[len(prefix):]]=reader.get_tensor(key)
    layer.load_state_dict(state,strict=True,assign=True)
    layer=layer.to(device='cuda',dtype=torch.bfloat16).eval()
    rotary=Qwen3_5TextRotaryEmbedding(cfg,device='cuda')
    saved=torch.load(PREVIOUS/'reference-layers-rank0.pt',map_location='cpu',weights_only=True,mmap=True)
    valid=saved['base_model.model.model.layers.2'].to('cuda')
    real_layer_output=saved['base_model.model.model.layers.3']
    original=ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',None)
    captures={};outputs={};timings={}
    mode=None
    def observed(module,query,key,value,attention_mask,**kwargs):
        result=original(module,query,key,value,attention_mask,**kwargs)
        captures[mode]=dict(q=sample(query,2),k=sample(key,2),v=sample(value,2),output=sample(result[0]),
            q_shape=list(query.shape),mask_shape=list(attention_mask.shape),mask_dtype=str(attention_mask.dtype))
        return result
    ALL_ATTENTION_FUNCTIONS.register('sdpa',observed)
    try:
        with torch.no_grad():
            for mode,length in [('reference',32320),('candidate',4160)]:
                h=torch.zeros((4,length,cfg.hidden_size),dtype=torch.bfloat16,device='cuda')
                mask=torch.zeros((4,length),dtype=torch.long,device='cuda')
                for row,n in enumerate(LENGTHS):
                    h[row,length-n:length-n+512]=valid[row]
                    mask[row,-n:]=1
                positions=(mask.cumsum(-1)-1).clamp_min(0)
                causal=create_causal_mask(config=cfg,inputs_embeds=h,attention_mask=mask,
                    past_key_values=None,position_ids=positions)
                embeddings=rotary(h,positions)
                start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=layer(hidden_states=h,position_embeddings=embeddings,
                                 attention_mask=causal,position_ids=positions,past_key_values=None)
                torch.cuda.synchronize()
                outputs[mode]=sample(output);timings[mode]=time.perf_counter()-start
                del h,mask,positions,causal,embeddings,output
                print(mode,'captured',flush=True)
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    ref=captures['reference'];cand=captures['candidate']
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        fp32=original(layer.self_attn,ref['q'].cuda().float(),ref['k'].cuda().float(),ref['v'].cuda().float(),
                      None,dropout=0.0,scaling=layer.self_attn.scaling)[0].cpu()
        bf16=original(layer.self_attn,ref['q'].cuda(),ref['k'].cuda(),ref['v'].cuda(),
                      None,dropout=0.0,scaling=layer.self_attn.scaling)[0].cpu()
    result=dict(scope=__doc__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_path=str(MODEL),owner_layer=type(layer).__name__,owner_attention=str(original),
        parameter_dtypes=sorted({str(p.dtype) for p in layer.parameters()}),
        qkv_comparison={k:difference(cand[k],ref[k]) for k in ['q','k','v']},
        attention_output_difference=difference(cand['output'],ref['output']),
        error_vs_fp32_math={k:difference(v,fp32) for k,v in [('reference',ref['output']),('candidate',cand['output']),('bf16_math',bf16)]},
        full_layer_difference=difference(outputs['candidate'],outputs['reference']),
        replay_vs_original_valid_prefix=difference(outputs['reference'],real_layer_output),
        timing_seconds=timings,shapes={m:{k:v for k,v in c.items() if not isinstance(v,torch.Tensor)} for m,c in captures.items()},
        limits='Only first 512 valid tokens per row; remaining hidden positions zero-filled, future positions cannot influence this causal prefix. No tolerance pass/fail inferred from these diagnostics.')
    torch.save(dict(captures=captures,outputs=outputs,fp32=fp32,bf16=bf16),OUT/'attention-operands.pt')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
