"""Compare the reorder on the saved noncoincident finite endpoint block.

This supplements the original FLA derivative-limit assertions. Its native
assert_close calculator/thresholds are reused for a finite equivalence check,
not described as an official test of nonzero DT attribution accuracy.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import time

import torch
import fla.utils
from finite_fla_gpu import make_compiled_finite_pullback
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders
from check_fla_early_scale import ref,represented_seed,restored


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('operands','precast','official','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    assert ref(args.official)['sha256']=='35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    assert not fla.utils.FLA_CI_ENV
    tree=ast.parse(args.official.read_bytes())
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_chunk')
    thresholds={n.args[0].value:n.args[3].value for n in ast.walk(test)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='assert_close'}
    saved=torch.load(args.operands,map_location='cpu',weights_only=False,mmap=True)
    before=torch.load(args.precast,map_location='cpu',weights_only=False,mmap=True)
    endpoints,old_seed,scale=saved['args']
    bad=(~torch.isfinite(old_seed)).nonzero();assert len(bad)==1
    start=int(bad[0,1])//64*64;end=min(start+64,old_seed.shape[1])
    paired={k:v[:,start//64:start//64+1].contiguous().cuda() if k=='h' else v[:,start:end].contiguous().cuda()
        for k,v in endpoints.items()}
    seed32=before['mo'][:,start:end,8:16].cuda()
    finite=average_memory_endpoint_orders(make_compiled_finite_pullback(
        reuse_scalar_products=False,dynamic_shapes=True,
        compiler_options={'triton.cudagraphs':False,'max_autotune':False}))
    result=dict(scope=__doc__,sources=[ref(p) for p in (args.operands,args.precast,args.official,Path(__file__))],
        start=start,tokens=end-start,scale=scale,calls=[],checks=[],production_changed=False,
        operations=dict(model=0,DT=0,optimizer=0),noncoincident=True)
    outputs=[]
    for name,seed,factor in (('accepted_range_base',seed32,scale),('early_scale',seed32*scale,1.0)):
        do,exponent=represented_seed(seed.to(before['output_dtype']),torch.float16)
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.no_grad():
            value={k:restored(v,exponent) for k,v in finite(paired,do,factor).items()}
        torch.cuda.synchronize()
        result['calls'].append(dict(name=name,seconds=time.perf_counter()-tick,
            native_seed_maxabs=float(do.abs().max()),exponent=exponent.cpu().flatten().tolist()))
        outputs.append(value)
    for name,key in zip(('q','k','v','beta','g'),('dq','dk','dv','db','dg')):
        expected,actual=outputs[0][name],outputs[1][name]
        check=dict(name=name,threshold=thresholds[key],normalized_error=float(fla.utils.get_err_ratio(expected,actual)),
            max_abs=float(fla.utils.get_abs_err(expected,actual)),nonfinite=int((~torch.isfinite(actual)).sum()))
        try:
            fla.utils.assert_close('noncoincident reorder '+name,expected,actual,thresholds[key])
            check['status']='passed'
        except AssertionError as error:
            check.update(status='failed',error=str(error))
        result['checks'].append(check)
    result.update(status='passed' if all(c['status']=='passed' for c in result['checks']) else 'failed',
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['status']=='passed','Keep the accepted base if finite equivalence exceeds original FLA tolerance'


if __name__=='__main__':
    main()
