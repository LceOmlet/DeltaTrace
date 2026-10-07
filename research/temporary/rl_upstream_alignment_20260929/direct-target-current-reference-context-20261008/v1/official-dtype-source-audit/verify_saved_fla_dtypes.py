"""Replay recorded GDN operands in both native dtypes, with original FLA checks.

This is an operand diagnostic. It does not change production dtype, reference,
thresholds or finite formulas. Coincident DT endpoints test the local derivative
limit; they do not certify an entire nonzero finite attribution.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from finite_fla_gpu import make_compiled_finite_pullback
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders
from verify_official_kernel_tolerances import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    source = args.sources / 'test_gated_delta_v041.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    assert not fla.utils.FLA_CI_ENV
    reference_owner = load('saved_operand_fla_reference', source)
    saved = torch.load(args.operands, map_location='cpu', weights_only=True, mmap=True)
    endpoints = saved['endpoints']
    names = ('q', 'k', 'v', 'beta', 'raw_g')
    # Four actual factual rows from the recorded B8 interleaved endpoints.
    original = [endpoints[name][1::2].cuda().detach() for name in names]
    initial = endpoints['h'][1::2, 0].cuda().float()
    scale = saved['scale']
    upstream = (saved['upstream'].cuda() if 'upstream' in saved
                else torch.randn_like(original[2]))
    result = dict(scope=__doc__, source=str(args.operands), cases=[],
                  original_operand_dtypes={name:str(value.dtype) for name,value in zip(names, original)},
                  shape=list(original[0].shape), initial_state_dtype=str(initial.dtype),
                  recorded_initial_state_dtype=str(endpoints['h'].dtype),
                  recorded_upstream_dtype=str(upstream.dtype),
                  original_assertion_source=str(source),
                  upstream_source='recorded finite consumer' if 'upstream' in saved else 'seeded diagnostic')
    finite_owner = average_memory_endpoint_orders(make_compiled_finite_pullback(dynamic_shapes=True))
    stage = importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd

    def check(name, expected, actual, threshold):
        record = dict(name=name, expected_dtype=str(expected.dtype), actual_dtype=str(actual.dtype),
                      threshold=threshold, rms_ratio=float(fla.utils.get_err_ratio(expected,actual)),
                      max_abs=float(fla.utils.get_abs_err(expected,actual)))
        try:
            fla.utils.assert_close(name, expected, actual, threshold)
            record['status'] = 'passed'
        except AssertionError as exc:
            record.update(status='failed', error=str(exc))
        return record

    for dtype in (torch.bfloat16, torch.float16):
        started = time.perf_counter()
        inputs = [value.to(torch.float32 if name == 'raw_g' else dtype).detach().requires_grad_()
                  for name,value in zip(names,original)]
        q,k,v,beta,g = inputs
        captured = {}
        def capture(frame, event, value):
            if frame.f_code is stage.__code__ and event == 'return' and value is not None:
                for name in ('q','k','v','g','beta','A','w','v_new','h'):
                    captured[name] = frame.f_locals[name].detach().repeat_interleave(2,0)
        assert sys.getprofile() is None
        sys.setprofile(capture)
        try:
            out,_ = chunk_gated_delta_rule(q=q,k=k,v=v,beta=beta,g=g,scale=scale,
                initial_state=initial,output_final_state=False,use_qk_l2norm_in_kernel=False)
        finally:
            sys.setprofile(None)
        captured['raw_g'] = g.detach().repeat_interleave(2,0)
        # Production passes native_mo, after the norm-input and q dtype casts.
        # Preserve its captured values/dtype on the actual FP16 case. BF16 is
        # an explicit same-operands cast comparison, not the deployed path.
        do = upstream.to(dtype)
        native_grad = torch.autograd.grad(out, inputs, do)
        ref,_ = reference_owner.recurrent_gated_delta_rule_ref(q=q,k=k,v=v,beta=beta,g=g,
            scale=scale,initial_state=initial,output_final_state=False)
        ref_grad = torch.autograd.grad(ref,inputs,do.float())
        with torch.no_grad():
            # Production finite path partitions heads by8. Call that same owner.
            pieces = [finite_owner({name:value[:,:,head:head+8].contiguous() for name,value in captured.items()},
                                   do[:,:,head:head+8].contiguous(), scale)
                      for head in range(0,q.shape[2],8)]
            finite = {name:torch.cat([piece[name] for piece in pieces],2) for name in pieces[0]}
        case = dict(dtype=str(dtype), operand_dtypes={name:str(x.dtype) for name,x in zip(names,inputs)},
                    native_upstream_dtype=str(do.dtype), finite_upstream_dtype=str(do.dtype),
                    reference_upstream_dtype=str(do.float().dtype),
                    initial_state_dtype=str(initial.dtype), initial_state_nonzero=bool(initial.count_nonzero()),
                    input_cast_max_abs={name:float((x.float()-y.detach().float()).abs().max())
                                        for name,x,y in zip(names,original,inputs)}, checks=[])
        case['checks'].append(check('native o',ref,out,.005))
        for name, expected, actual in zip(('q','k','v','beta','g'),ref_grad,native_grad):
            threshold = .02 if name in ('beta','g') else .008
            case['checks'].append(check('native d'+name,expected,actual,threshold))
            case['checks'].append(check('finite d'+name,expected,finite[name],threshold))
        torch.cuda.synchronize()
        case['seconds'] = time.perf_counter()-started
        case['status'] = 'passed' if all(item['status']=='passed' for item in case['checks']) else 'failed'
        result['cases'].append(case)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(case),flush=True)
        del inputs,q,k,v,beta,g,captured,out,ref,native_grad,ref_grad,pieces,finite
    result['status'] = 'completed_actual_operand_comparison'
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
