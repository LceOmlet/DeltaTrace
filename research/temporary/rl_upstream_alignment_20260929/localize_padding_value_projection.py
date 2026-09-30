"""Locate dtype/shape effects using the original Qwen norm and nn.Linear only."""
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

out=Path(os.environ['PADDING_DIAGNOSTIC_DIR'])
model=Path(os.environ['MODEL_PATH'])
cfg=AutoConfig.from_pretrained(model,local_files_only=True).text_config
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
prefix='model.language_model.layers.3.'
def weight(name):
    key=prefix+name
    with safe_open(model/index[key],framework='pt',device='cpu') as f:
        return f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16)
norm=Qwen3_5RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
norm.weight=torch.nn.Parameter(weight('input_layernorm.weight'),requires_grad=False)
proj=torch.nn.Linear(cfg.hidden_size,cfg.num_key_value_heads*cfg.head_dim,bias=False,
                     device='cuda',dtype=torch.bfloat16)
proj.weight=torch.nn.Parameter(weight('self_attn.v_proj.weight'),requires_grad=False)
saved=torch.load(out.parent/'layer-localization-v2/reference-layers-rank0.pt',
                 map_location='cpu',weights_only=True,mmap=True)
valid=saved['base_model.model.model.layers.2'].cuda();lengths=[512,1024,2048,4096]
def sample(value):
    return torch.stack([value[i,value.shape[1]-n:value.shape[1]-n+512] for i,n in enumerate(lengths)]).detach().cpu().clone()
def difference(a,b):
    d=a.float()-b.float()
    return dict(exact=bool(torch.equal(a,b)),max_abs=float(d.abs().max()),
                rms=float(d.square().mean().sqrt()),reference_rms=float(b.float().square().mean().sqrt()))
captures={}
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for name,length in [('reference',32320),('candidate',4160)]:
        h=torch.zeros((4,length,cfg.hidden_size),dtype=torch.bfloat16,device='cuda')
        for row,n in enumerate(lengths):h[row,length-n:length-n+512]=valid[row]
        normalized=norm(h);projected=proj(normalized)
        captures[name]=dict(input=sample(h),normalized=sample(normalized),projected=sample(projected))
        del h,normalized,projected
with torch.no_grad():
    torch.backends.cuda.matmul.allow_tf32=False
    fp32=torch.nn.functional.linear(captures['reference']['normalized'].cuda().float(),
                                   proj.weight.float()).cpu()
old=torch.load(out.parent/'attention-replay/attention-operands.pt',map_location='cpu',weights_only=True)
result=dict(scope=__doc__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    parameter_dtype=str(proj.weight.dtype),input_dtype=str(valid.dtype),reference_dtype=str(fp32.dtype),
    fp32_reference_tf32=False,operation='original Qwen3_5RMSNorm then original torch.nn.Linear',
    between_layouts={k:difference(captures['candidate'][k],captures['reference'][k]) for k in ['input','normalized','projected']},
    error_vs_fp32={k:difference(v['projected'],fp32) for k,v in captures.items()},
    matches_previous_attention_v={k:difference(v['projected'],old['captures'][k]['v'].transpose(1,2).reshape(4,512,-1)) for k,v in captures.items()},
    scope_limit='Diagnostic only, no invented pass threshold. Same saved first512-token causal prefixes; no full-model reload or update.')
torch.save(dict(captures=captures,fp32=fp32),out/'operands.pt')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
