"""Read public varlen output layout on saved actual FA operands; no new tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input
from transformers.modeling_flash_attention_utils import _upad_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    path=root/'receipts/upstream-alignment-20260929/actual-dt-layer3-boundaries.pt'
    fa=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['fa']
    # Use the original factual endpoint arrays. There is no fabricated u,
    # alternate target, model rollout or action-range change in this probe.
    q,k,v=[fa['operands'][name].to('cuda').transpose(1,2)
           for name in ('q0','k0','v0')]
    batch,query_length=q.shape[:2]
    mask=torch.ones((batch,k.shape[1]),device=q.device,dtype=torch.long)
    torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.no_grad():
        pq,pk,pv,indices,(cuq,cuk),(maxq,maxk)=_upad_input(q,k,v,mask,query_length,unpad_input)
        out,lse,unused=flash_attn_varlen_func(pq,pk,pv,cuq,cuk,maxq,maxk,
            dropout_p=0.,softmax_scale=fa['scale'],causal=True,return_attn_probs=True)
        actual=pad_input(out,indices,batch,query_length)
        dense,dense_lse,dense_unused=flash_attn_func(q,k,v,
            dropout_p=0.,softmax_scale=fa['scale'],causal=True,return_attn_probs=True)
    torch.cuda.synchronize()
    result=dict(scope=__doc__,status='public_interface_measured_not_numerical_acceptance',
        saved_artifact=dict(path=str(path),sha256='0ade21d748c0fa37bd46a08ee3c1452d6cb298dac36b0fed6ad7c7be4b9978aa'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_operand_shapes=dict(q=list(q.shape),k=list(k.shape),v=list(v.shape)),
        actual_public_return=dict(out=list(out.shape),LSE=list(lse.shape),LSE_dtype=str(lse.dtype),
            dense_LSE=list(dense_lse.shape),cu_q=cuq.cpu().tolist(),cu_k=cuk.cpu().tolist(),
            max_seqlen_q=int(maxq),max_seqlen_k=int(maxk)),
        descriptive_differences_not_pass_threshold=dict(
            native_varlen_vs_native_dense_output_max_abs=float((actual-dense).abs().max())),
        seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        model_loads=0,checkpoint_loads=0,production_changes=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
