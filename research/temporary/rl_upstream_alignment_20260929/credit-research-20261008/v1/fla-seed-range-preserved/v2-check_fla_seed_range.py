"""Diagnose the saved real FP16 seed overflow with unchanged original FLA calls.

The numerical identity is F_E(2**e*u)=2**e*F_E(u): each original
adjoint/coefficient stage is linear in its incoming cotangent for fixed
endpoints, including the symmetric average. This is a range representation
experiment, not credit correction, clipping, a changed finite allocation,
an official tolerance certificate, or a deployed owner patch.
"""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time
import torch


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--operands',type=Path,required=True)
    parser.add_argument('--precast',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    source=json.loads(args.source.read_bytes())
    record=dict(scope=__doc__,source=dict(path=str(args.source),sha256=sha(args.source)),
        operands=dict(path=str(args.operands),sha256=sha(args.operands)),
        precast=dict(path=str(args.precast),sha256=sha(args.precast)),
        script=dict(path=__file__,sha256=sha(__file__)),production_modified=False,
        model_calls=0,DT_calls=0,optimizer=0,official_tolerance_claim=False)
    saved=torch.load(args.operands,map_location='cpu',weights_only=False)
    before=torch.load(args.precast,map_location='cpu',weights_only=False)
    endpoints,old_seed,scale=saved['args']
    assert saved['layer']==before['layer']==4
    # The first 8-head call was finite; the second 8-head call was saved.
    group=before['mo'][:,:,8:16].to(before['output_dtype'])
    assert group.shape==old_seed.shape and old_seed.dtype==torch.float16
    cast=group.to(old_seed.dtype)
    record['precast_identity']=dict(exactly_matches_saved_FP16_seed=torch.equal(cast,old_seed),
        differing_elements=int((cast!=old_seed).sum()),precast_nonfinite=int((~torch.isfinite(group)).sum()),
        FP16_nonfinite=int((~torch.isfinite(cast)).sum()),maxabs=float(group.abs().max()))
    assert record['precast_identity']['exactly_matches_saved_FP16_seed']
    bad=(~torch.isfinite(cast)).nonzero()
    record['overflow_values']=[dict(position=idx,precast=float(group[tuple(idx)]),FP16=float(cast[tuple(idx)]))
                               for idx in bad[:16].tolist()]
    module=importlib.import_module('finite_fla_gpu')
    symmetric=importlib.import_module('profiles.qwen35_gdn_symmetric')
    for name,owner in [('finite_fla_gpu',module),('symmetric',symmetric)]:
        record[name]=dict(path=owner.__file__,sha256=sha(owner.__file__))
    assert record['finite_fla_gpu']['sha256']=='f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6'
    assert record['symmetric']['sha256']=='dd6bbfff9aae4c679439af113a160e0bd8c4cf1b19bb17cab47a04ff7b0831d0'
    # The smallest integer power that fits the actual native dtype. Unit-range
    # normalization was unnecessarily strong and amplified subnormal loss.
    # This bound is torch.finfo's representation limit, not a tuned threshold.
    group=group.to('cuda').float()
    magnitude=group.abs().amax(dim=(1,3),keepdim=True)
    exponent=torch.ceil(torch.log2(magnitude/torch.finfo(torch.float16).max).clamp_min(0)).to(torch.int32)
    normalized=torch.ldexp(group,-exponent).to(torch.float16)
    record['normalization']=dict(exponents=exponent.cpu().flatten().tolist(),
        maxabs=float(normalized.abs().max()),nonfinite=int((~torch.isfinite(normalized)).sum()))
    del group,magnitude,cast,before
    endpoints={k:v.to('cuda') for k,v in endpoints.items()}
    callback=symmetric.average_memory_endpoint_orders(module.make_compiled_finite_pullback(
        reuse_scalar_products=False,dynamic_shapes=True,compiler_options={'triton.cudagraphs':False,'max_autotune':False}))
    outputs=[]
    for index in range(2):
        torch.cuda.synchronize();tick=time.perf_counter()
        # Second scale is a linearity diagnostic using the same saved endpoints.
        seed=normalized if index==0 else normalized*0.5
        value=callback(endpoints,seed,scale)
        restored={key:torch.ldexp(v.float(),exponent if v.ndim==4 else exponent[...,0])
                  for key,v in value.items()}
        if index:restored={k:v*2 for k,v in restored.items()}
        torch.cuda.synchronize()
        record.setdefault('calls',[]).append(dict(seconds=time.perf_counter()-tick,
            outputs={k:dict(nonfinite=int((~torch.isfinite(v)).sum()),maxabs=float(v.abs().max()))
                     for k,v in restored.items()}))
        outputs.append({k:v.cpu() for k,v in restored.items()})
        del value,restored,seed
    record['power_two_linearity']={k:dict(maxabs_difference=float((outputs[0][k]-outputs[1][k]).abs().max()),
        relative_l2=float((outputs[0][k]-outputs[1][k]).double().norm()/outputs[0][k].double().norm()))
        for k in outputs[0]}
    artifact=args.output.with_suffix('.pt');torch.save(dict(outputs=outputs,exponent=exponent.cpu()),artifact)
    record['artifact']=dict(path=str(artifact),sha256=sha(artifact),bytes=artifact.stat().st_size)
    record['peak_allocated']=torch.cuda.max_memory_allocated();record['peak_reserved']=torch.cuda.max_memory_reserved()
    record['status']='diagnostic_complete_not_accepted'
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='normalization'}))


if __name__=='__main__':main()
