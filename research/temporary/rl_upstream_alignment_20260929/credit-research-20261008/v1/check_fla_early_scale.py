"""Check the scale reorder with the installed FLA and its original assertions.

Reuse the actual overflow block, both actual operand dtypes, the saved nonzero
initial state, original reference recurrence, native kernels and finite owner.
No model, training, attribution estimator or tolerance is reimplemented.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from finite_fla_gpu import make_compiled_finite_pullback
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders


def ref(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=digest)


def represented_seed(seed, dtype):
    # Exact accepted representation, shared across all times of a head.
    exponent = None
    value = seed
    if dtype == torch.float16:
        magnitude = value.float().abs().amax((1,3),keepdim=True)
        exponent = torch.ceil(torch.log2(magnitude/torch.finfo(dtype).max).clamp_min(0)).int()
        value = torch.ldexp(value.float(),-exponent)
    return value.to(dtype),exponent


def restored(value, exponent):
    return value.float() if exponent is None else torch.ldexp(value.float(),exponent if value.ndim==4 else exponent[...,0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('operands','precast','official','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert ref(args.official)['sha256']=='35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    assert not fla.utils.FLA_CI_ENV
    spec = importlib.util.spec_from_file_location('official_fla_scale_reference',args.official)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    tree = ast.parse(args.official.read_bytes())
    test = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_chunk')
    thresholds = {n.args[0].value:n.args[3].value for n in ast.walk(test)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='assert_close'}
    saved = torch.load(args.operands,map_location='cpu',weights_only=False,mmap=True)
    before = torch.load(args.precast,map_location='cpu',weights_only=False,mmap=True)
    endpoints,old_seed,scale = saved['args']
    seed32 = before['mo'][:,:,8:16]
    assert seed32.dtype==torch.float32
    assert torch.equal(seed32.to(before['output_dtype']).to(old_seed.dtype),old_seed)
    bad = (~torch.isfinite(old_seed)).nonzero()
    assert len(bad)==1
    start = int(bad[0,1])//64*64
    end = min(start+64,seed32.shape[1])
    original = {name:endpoints[name][1::2,start:end].contiguous()
        for name in ('q','k','v','beta','raw_g')}
    initial = endpoints['h'][1::2,start//64].float().cuda()
    native_seed32 = seed32[:,start:end].cuda()
    result = dict(scope=__doc__,sources=[ref(p) for p in (args.operands,args.precast,args.official,Path(__file__))],
        official_thresholds=thresholds,scale=scale,capture_local_block_start=start,tokens=end-start,
        reference='Original FLA FP32 recurrence on exact native dtype operands, original unrounded FP32 cotangent. '
            'Finite coincident-endpoint derivative limit is checked; this is not a nonzero DT quality certificate.',
        production_changed=False,operations=dict(model=0,DT=0,optimizer=0),cases=[])
    finite = average_memory_endpoint_orders(make_compiled_finite_pullback(
        reuse_scalar_products=False,dynamic_shapes=True,
        compiler_options={'triton.cudagraphs':False,'max_autotune':False}))
    stage = importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd

    def check(name,key,expected,actual):
        value = dict(name=name,threshold=thresholds[key],
            normalized_error=float(fla.utils.get_err_ratio(expected,actual)),
            max_abs=float(fla.utils.get_abs_err(expected,actual)),nonfinite=int((~torch.isfinite(actual)).sum()))
        try:
            fla.utils.assert_close(name,expected,actual,thresholds[key])
            value['status']='passed'
        except AssertionError as error:
            value.update(status='failed',error=str(error))
        return value

    for dtype in (torch.float16,torch.bfloat16):
        tick = time.perf_counter()
        inputs = [original[name].cuda().to(torch.float32 if name=='raw_g' else dtype).detach().requires_grad_()
            for name in ('q','k','v','beta','raw_g')]
        reference_inputs = [v.detach().float().requires_grad_() for v in inputs]
        reference,_ = owner.recurrent_gated_delta_rule_ref(q=reference_inputs[0],k=reference_inputs[1],
            v=reference_inputs[2],beta=reference_inputs[3],g=reference_inputs[4],scale=scale,
            initial_state=initial,output_final_state=False)
        expected = torch.autograd.grad(reference,reference_inputs,native_seed32)
        new_scale = 1.0 if dtype==torch.float16 else scale
        seed = native_seed32*scale if dtype==torch.float16 else native_seed32
        do,exponent = represented_seed(seed.to(before['output_dtype']),dtype)
        captures = {}

        def capture(frame,event,value):
            if frame.f_code==stage.__code__ and event=='return' and value is not None:
                for name in ('q','k','v','g','beta','A','w','v_new','h'):
                    captures[name]=frame.f_locals[name].detach().repeat_interleave(2,0)

        assert sys.getprofile() is None
        sys.setprofile(capture)
        try:
            output,_ = chunk_gated_delta_rule(q=inputs[0],k=inputs[1],v=inputs[2],beta=inputs[3],g=inputs[4],
                scale=new_scale,initial_state=initial,output_final_state=False,use_qk_l2norm_in_kernel=False)
        finally:
            sys.setprofile(None)
        captures['raw_g']=inputs[4].detach().repeat_interleave(2,0)
        actual = [restored(x,exponent) for x in torch.autograd.grad(output,inputs,do)]
        with torch.no_grad():
            coefficients = {k:restored(v,exponent) for k,v in finite(captures,do,new_scale).items()}
        checks = [check('native scaled o','o',reference,output*scale if dtype==torch.float16 else output)]
        for name,key,want,got in zip(('q','k','v','beta','g'),('dq','dk','dv','db','dg'),expected,actual):
            checks.append(check('native reordered d'+name,key,want,got))
            checks.append(check('finite reordered d'+name,key,want,coefficients[name]))
        torch.cuda.synchronize()
        result['cases'].append(dict(dtype=str(dtype),seed_dtype=str(do.dtype),original_scale=scale,
            callback_scale=new_scale,seed_maxabs=float(do.abs().max()),seed_nonfinite=int((~torch.isfinite(do)).sum()),
            exponent=None if exponent is None else exponent.cpu().flatten().tolist(),checks=checks,
            seconds=time.perf_counter()-tick,status='passed' if all(c['status']=='passed' for c in checks) else 'failed'))
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['cases'][-1]),flush=True)
        del inputs,reference_inputs,reference,expected,output,actual,coefficients,captures,do
    result.update(status='passed' if all(c['status']=='passed' for c in result['cases']) else 'failed',
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='passed','Original official FLA tolerance failed; do not deploy'


if __name__=='__main__':
    main()
