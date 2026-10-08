"""Default owner parity and native convolution graph on saved real operands.

No model/DT/optimizer, no author-quality claim and no new numerical threshold.
"""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn,causal_conv1d_ref
from conditional_conv_windows import conditional_conv_windows,native_qkv
from check_conditional_conv_windows import official_tolerances,identity
from finite_fla_gpu import native_input_adjoints,mixed_coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('original_fla','original_query','original_conv','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    assert identity(args.original_fla)['sha256']=='f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6'
    original=json.loads((args.original_query/'result.json').read_bytes())
    conv=json.loads((args.original_conv/'result.json').read_bytes())
    spec=importlib.util.spec_from_file_location('preserved_original_fla',args.original_fla)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    raw=torch.load(original['operands']['path'],map_location='cpu',mmap=True,weights_only=False)
    first=original['block'][0];last=first+64
    endpoints={key:value[:,first//64:last//64] if key=='h' else value[:,first:last]
               for key,value in raw['args'][0].items()}
    endpoints={key:value.cuda().contiguous() for key,value in endpoints.items()}
    query=torch.load(original['cases'][0]['exact_artifact']['path'],map_location='cpu',mmap=True,weights_only=False)
    do=query['do'][:,:64].cuda().to(endpoints['q'].dtype).contiguous()
    scale=query['scale']
    result=dict(scope=__doc__,sources={n:identity(importlib.import_module(n).__file__)
                for n in ('conditional_conv_windows','conditional_gdn_context','finite_fla_gpu','qwen35_gdn_finite')},
                script=identity(__file__),cases=[],model_calls=0,DT_calls=0,optimizer=0,
                production_modified=False,whole_DT_repair_accepted=False)
    with torch.no_grad():
        adjoints=native_input_adjoints(endpoints,do,scale)
        left,ld=old.mixed_coefficients(endpoints,adjoints,scale,diagnostics=True)
        right,rd=mixed_coefficients(endpoints,adjoints,scale,diagnostics=True)
        checks={key:bool(torch.equal(left[key],right[key])) for key in left}
        checks.update({key:bool(torch.equal(ld[key],rd[key])) for key in ld})
        assert all(checks.values()),checks
        result['unchanged_fla_default_bitwise']=checks
    del raw,endpoints,query,do,adjoints,left,right,ld,rd
    original_input=torch.load(conv['operands']['path'],map_location='cpu',mmap=True,weights_only=False)['tensors']
    assert identity(conv['operands']['path'])['sha256']==conv['operands']['sha256']
    for case in conv['cases']:
        tick=time.perf_counter();dtype=getattr(torch,case['dtype'].split('.')[-1]);T=199
        assert identity(case['exact_artifact']['path'])['sha256']==case['exact_artifact']['sha256']
        saved=torch.load(case['exact_artifact']['path'],map_location='cpu',mmap=True,weights_only=False)
        x,x0,h,w=[saved[key].cuda() for key in ('factual','replacement','initial','weight')]
        with torch.no_grad():
            default=conditional_conv_windows(causal_conv1d_fn,x,x0,w,initial=h)
            norm=native_qkv(default['output'],key_heads=16,value_heads=32,key_dim=128,value_dim=128)
            parity={key:bool(torch.equal(value.cpu(),saved['windows'][key])) for key,value in default.items()}
            parity.update({key:bool(torch.equal(value.cpu(),saved['qkv'][key])) for key,value in norm.items()})
            assert all(parity.values()),parity
            pack=conditional_conv_windows(causal_conv1d_fn,x,x0,w,initial=h,retain_pre_graph=True)
            graph_parity={key:bool(torch.equal(pack[key],default[key])) for key in ('pre','output','valid')}
            assert all(graph_parity.values())
            # Actual saved cotangent; smallest integer power of two only
            # keeps this diagnostic's FP16 representation finite. Both
            # native/reference calls consume exactly the same operand.
            seed=original_input['finite_seed'][1::2,:,4:4+T].cuda().float()
            assert bool(torch.isfinite(seed).all())
            exponent=torch.ceil(torch.log2(seed.abs().amax()/torch.finfo(dtype).max).clamp_min(0)).int()
            seed=torch.ldexp(seed,-exponent).to(dtype)
            windows=torch.stack([torch.nn.functional.pad(seed[:,:,lag:],(0,lag)) for lag in range(4)])
            window_seed=windows.permute(1,3,2,0).reshape(4*T,8192,4).contiguous()
        with torch.enable_grad():
            gx,=torch.autograd.grad(pack['native_pre'],pack['native_input'],window_seed)
            full=x.detach().transpose(1,2).contiguous().transpose(1,2).requires_grad_(True)
            y=causal_conv1d_fn(full,w,initial_states=h,activation=None)
            expected,=torch.autograd.grad(y,full,seed)
            full_ref=full.detach().clone().requires_grad_(True)
            yref=causal_conv1d_ref(full_ref,w.float(),initial_states=h,activation=None)
            expected_ref,=torch.autograd.grad(yref,full_ref,seed)
        actual=gx[:,:,0].reshape(4,T,8192).transpose(1,2)
        rtol,atol=official_tolerances(Path(conv['official_test']['path']),dtype)
        assert torch.allclose(expected,expected_ref,rtol=rtol,atol=atol)
        assert torch.allclose(actual,expected_ref,rtol=rtol,atol=atol)
        torch.cuda.synchronize()
        row=dict(dtype=case['dtype'],default_forward_bitwise=parity,graph_forward_bitwise=graph_parity,
            graph_input_gradient_original_tolerance=dict(rtol=rtol,atol=atol,passed=True),
            packed_full_dx_bitwise=bool(torch.equal(actual,expected)),
            packed_full_dx_max_abs=float((actual.float()-expected.float()).abs().max()),
            packed_reference_dx_max_abs=float((actual.float()-expected_ref.float()).abs().max()),
            original_upstream_exponent=int(exponent),nonzero_native_seed=int(seed.count_nonzero()),
            seconds=time.perf_counter()-tick)
        result['cases'].append(row)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
        del saved,x,x0,h,w,default,norm,pack,seed,windows,window_seed,gx,full,y,expected,full_ref,yref,expected_ref,actual
    result.update(phase='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
