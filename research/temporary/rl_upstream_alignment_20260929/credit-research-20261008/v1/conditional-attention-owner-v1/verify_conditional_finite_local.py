"""Nonzero local-identity diagnostic on the original saved B4 attention operands.

All first 64 eligible source rows are retained, without inspecting effects.
This checks a primitive's semantics, not population attribution quality. The
pinned author attention_ref owns every reference attention evaluation. There
is no invented nonzero-finite tolerance and no model or optimizer operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from flash_attn import flash_attn_func
from vendor_fa_finite_bf16_d256 import RightPaddedLengths, VendorFAFiniteP1BF16D256
from verify_official_kernel_tolerances import load


def quantiles(values):
    values=values.detach().double().cpu().flatten()
    finite=values[torch.isfinite(values)]
    result=dict(count=values.numel(),finite_count=finite.numel())
    if finite.numel():
        result.update(zip(('min','p25','median','p75','p95','max'),
            torch.quantile(finite,torch.tensor([0,.25,.5,.75,.95,1],dtype=torch.float64)).tolist()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operands',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--finite-library',type=Path,required=True)
    parser.add_argument('--finite-library-sha256',required=True)
    parser.add_argument('--composition',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8)
    reference_path=args.sources/'test_flash_attn_v263.py'
    reference_sha=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert reference_sha=='a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
    reference=load('nonzero_conditional_original_reference',reference_path)
    saved=torch.load(args.operands,map_location='cpu',weights_only=True,mmap=True)
    fa=saved['fa'];a=saved['attention_values']
    start=max(fa['coefficient_starts'])
    count=min(64,min(fa['lengths'])-start)
    end=start+count
    query_start=fa['query_start']
    offset=start-query_start
    assert count>0 and offset>=0
    # Keep the original native-aligned query boundary and original coefficient
    # cut. The owner's RightPaddedLengths reserves the 64-token alignment.
    q0,q1=[a['dense_q'][i::2,:offset+count].cuda().contiguous() for i in (0,1)]
    k0,k1=[a['dense_k'][i::2,:end].cuda().contiguous() for i in (0,1)]
    v0,v1=[a['dense_v'][i::2,:end].cuda().contiguous() for i in (0,1)]
    upstream=fa['operands']['u'][:,:,:offset+count].transpose(1,2).cuda().contiguous()
    # Owner ABI takes FP32 U and performs its original BF16 storage conversion.
    # Local reference and diagonal UV must use the values the kernel consumes.
    u=upstream.to(q1.dtype).contiguous()
    b,t,h,d=q1.shape;kh=k1.shape[2]
    scale=fa['scale']
    assert scale==d**-.5, 'This saved reference call uses the original default scale.'
    def diagonal(q,k):
        grouped=q.float().reshape(b,t,kh,h//kh,d)
        return (grouped*k[:,query_start:end].float().unsqueeze(3)).sum(-1).reshape(b,t,h).transpose(1,2).contiguous()
    own0=diagonal(q0,k0)*scale
    own1=diagonal(q1,k0)*scale
    own_uv0=diagonal(u,v0)
    print('phase=public_strict_past_FA count='+str(count),flush=True)
    with torch.no_grad():
        layout=RightPaddedLengths([end]*b,end,q1.device,coefficient_starts=[start]*b,query_start=query_start)
        composition=None
        composition_seconds=None
        # Native bottom-right causal alignment: query i sees keys strictly
        # before its corresponding logical query position query_start+i.
        if args.composition:
            from qwen35_decoder_finite import FiniteBoundaryOps
            from conditional_attention_endpoints import prepare_conditional_endpoints
            boundaries=FiniteBoundaryOps(True,dynamic_shapes=True)
            captures={name:torch.stack((left,right),1).flatten(0,1).transpose(1,2)
                for name,left,right in [('query',q0,q1),('key',k0,k1),('value',v0,v1)]}
            torch.cuda.synchronize();tick=time.perf_counter()
            composition=prepare_conditional_endpoints(captures,layout,scale,boundaries)
            compiled_uv=boundaries.attention_own_uv(upstream.transpose(1,2),v0.transpose(1,2),composition['query_starts'])
            torch.cuda.synchronize();composition_seconds=time.perf_counter()-tick
            conditional_lse0=composition['lse0']
            diagonal_diffs=dict(q0k0=float((composition['own_q0k0']-own0).abs().max()),
                q1k0=float((composition['own_q1k0']-own1).abs().max()),
                uv0=float((compiled_uv-own_uv0).abs().max()))
            own0,own1,own_uv0=composition['own_q0k0'],composition['own_q1k0'],compiled_uv
        else:
            _,past_lse,_=flash_attn_func(q0,k1[:,:-1],v1[:,:-1],causal=True,
                softmax_scale=scale,deterministic=True,return_attn_probs=True)
            conditional_lse0=torch.logaddexp(past_lse,own0)
        ops=dict(q0=q0.transpose(1,2),q1=q1.transpose(1,2),
            k0=k0.transpose(1,2),k1=k1.transpose(1,2),
            v0=v0.transpose(1,2),v1=v1.transpose(1,2),u=upstream.transpose(1,2),
            lse0=conditional_lse0,lse1=conditional_lse0,
            own_q0k0=own0,own_q1k0=own1,own_uv0=own_uv0)
        owner=VendorFAFiniteP1BF16D256(args.finite_library,args.finite_library_sha256)
        torch.cuda.synchronize();started=time.perf_counter()
        finite=owner(ops,scale,layout,conditional=True)
        torch.cuda.synchronize();kernel_seconds=time.perf_counter()-started
        predicted=(finite['dq'].transpose(1,2).double()*(q1.double()-q0.double())).sum((2,3))
        for name,factual,deleted in [('dk',k1,k0),('dv',v1,v0)]:
            coefficient=finite[name].double().reshape(b,kh,h//kh,t,d)
            delta=(factual[:,query_start:end].double()-deleted[:,query_start:end].double()).permute(0,2,1,3)
            predicted+=(coefficient*delta.unsqueeze(2)).sum((1,2,4))
        predicted=predicted[:,offset:offset+count]
        print('phase=original_reference_local_deletions rows='+str(b*count),flush=True)
        # upcast=False preserves FP64 operands in the original reference.
        # This is a semantic diagnostic, not a replacement official assertion.
        qf,kf,vf,ur=[x.double() for x in (q1,k1,v1,u)]
        factual,_=reference.attention_ref(qf,kf,vf,causal=True,upcast=False)
        exact=[];endpoint_errors=[];endpoint_relative=[]
        reference_started=time.perf_counter()
        for i in range(count):
            qc=qf.clone();kc=kf.clone();vc=vf.clone()
            qc[:,offset+i]=q0[:,offset+i].double()
            kc[:,start+i]=k0[:,start+i].double()
            vc[:,start+i]=v0[:,start+i].double()
            deleted,_=reference.attention_ref(qc,kc,vc,causal=True,upcast=False)
            exact.append(((factual-deleted)*ur).sum((1,2,3)))
            if composition is not None:
                endpoint=deleted[:,offset+i]
                difference=composition['attention0'][:,offset+i].double()-endpoint
                endpoint_errors.append(difference.abs().amax(dim=(1,2)))
                endpoint_relative.append(difference.flatten(1).norm(dim=1)/endpoint.flatten(1).norm(dim=1))
        exact=torch.stack(exact,1)
        torch.cuda.synchronize();reference_seconds=time.perf_counter()-reference_started
        error=predicted-exact
        nonfinite={name:int((~torch.isfinite(value[:,:,offset:offset+count])).sum()) for name,value in finite.items()}
        result=dict(scope=__doc__,status='Diagnostic completed; no new finite-effect pass threshold',
            actual_shape=dict(q=list(q1.shape),k=list(k1.shape),u=list(u.shape)),
            dtypes=dict(upstream_ABI=str(upstream.dtype),kernel_upstream=str(u.dtype),reference='FP64 of exact BF16 stored operands'),
            original_query_start=fa['query_start'],source_start=start,source_count=count,
            selected='Complete first 64 eligible rows of all four original operand pairs; no effect-based selection.',
            original_reference=dict(path=str(reference_path),sha256=reference_sha,dtype='FP64 via original upcast=False'),
            finite_library=str(args.finite_library),finite_library_sha256=args.finite_library_sha256,
            predicted=predicted.cpu().tolist(),reference=exact.cpu().tolist(),error=error.cpu().tolist(),
            abs_error=quantiles(error.abs()),reference_abs=quantiles(exact.abs()),
            relative_L2=float(error.norm()/exact.norm()) if float(exact.norm()) else None,
            sign_crossings=int(((predicted*exact)<0).sum()),nonfinite=nonfinite,
            finite_kernel_seconds=kernel_seconds,reference_seconds=reference_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            model_calls=0,optimizer_steps=0,
            limitations=['Not a frozen-collection sample or a method-quality result.',
                'Original FA derivative assertions remain in the separate unchanged receipt.',
                'No tolerance for this new nonzero finite rule is attributed to FA.'])
        if composition is not None:
            result['composition']=dict(used=True,cold_compiled_helpers_and_public_FA_seconds=composition_seconds,
                diagonal_max_difference=diagonal_diffs,
                conditional_endpoint_max_abs_by_row=quantiles(torch.stack(endpoint_errors)),
                conditional_endpoint_relative_L2_by_row=quantiles(torch.stack(endpoint_relative)),
                conditional_endpoint_dtype=str(composition['attention0'].dtype),
                scope='Original public varlen FA/unpad/pad plus finite owner compiled diagonal and own-output combination. No runner or model operation.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('predicted','reference','error')},indent=2),flush=True)


if __name__=='__main__':main()
