"""Check the real overflow block with original FLA reference and assertions.

Only the incoming cotangent's power-of-two representation changes. The saved
native operands and original recurrence/reference/assert_close are reused.
Coincident endpoints check the derivative limit, not nonzero DT accuracy.
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


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('operands', 'precast', 'official', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert sha(args.official) == '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    assert not fla.utils.FLA_CI_ENV
    spec = importlib.util.spec_from_file_location('original_fla_test_range', args.official)
    reference_owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_owner)
    # Read the actual original test's tolerances, without redefining them here.
    tree = ast.parse(args.official.read_bytes())
    test = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_chunk')
    thresholds = {n.args[0].value: n.args[3].value for n in ast.walk(test)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'assert_close'}
    saved = torch.load(args.operands, map_location='cpu', weights_only=False, mmap=True)
    before = torch.load(args.precast, map_location='cpu', weights_only=False, mmap=True)
    endpoints, old_seed, scale = saved['args']
    seed = before['mo'][:, :, 8:16].to(before['output_dtype'])
    assert torch.equal(seed.to(old_seed.dtype), old_seed)
    bad = (~torch.isfinite(old_seed)).nonzero()
    assert bad.shape[0] == 1
    start = int(bad[0, 1])//64*64
    end = min(start+64, seed.shape[1])
    assert endpoints['h'].shape[1] == (seed.shape[1]+63)//64
    original = {name: endpoints[name][1::2, start:end].contiguous()
                for name in ('q', 'k', 'v', 'beta', 'raw_g')}
    initial = endpoints['h'][1::2, start//64].float().cuda()
    native_output_seed = seed[:, start:end].float().cuda()
    exponent = torch.ceil(torch.log2(seed.float().abs().amax((1, 3), keepdim=True)
                                     /torch.finfo(torch.float16).max).clamp_min(0)).int().cuda()
    result = dict(scope=__doc__, script=dict(path=__file__, sha256=sha(__file__)),
                  original_test=dict(path=str(args.official), sha256=sha(args.official), thresholds=thresholds),
                  source_operands=dict(path=str(args.operands), sha256=sha(args.operands)),
                  source_precast=dict(path=str(args.precast), sha256=sha(args.precast)),
                  capture_local_block_start=start, block_tokens=end-start,
                  overflow_position=bad[0].tolist(), exponents=exponent.cpu().flatten().tolist(),
                  reference_input_precision='FP32 copies of the exact native low-precision operands; avoids gradient output cast overflow',
                  reference_seed_precision='Original BF16 norm-output values converted to FP32',
                  cases=[], production_modified=False, model_calls=0, DT_calls=0, optimizer=0)
    finite_owner = average_memory_endpoint_orders(make_compiled_finite_pullback(
        reuse_scalar_products=False, dynamic_shapes=True,
        compiler_options={'triton.cudagraphs': False, 'max_autotune': False}))
    stage = importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd

    def check(name, key, expected, actual):
        record = dict(name=name, threshold=thresholds[key],
                      normalized_error=float(fla.utils.get_err_ratio(expected, actual)),
                      max_abs=float(fla.utils.get_abs_err(expected, actual)),
                      nonfinite=int((~torch.isfinite(actual)).sum()))
        try:
            fla.utils.assert_close(name, expected, actual, thresholds[key])
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='failed', error=str(error))
        return record

    for dtype in (torch.float16, torch.bfloat16):
        tick = time.perf_counter()
        inputs = [original[name].cuda().to(torch.float32 if name == 'raw_g' else dtype)
                  .detach().requires_grad_() for name in ('q', 'k', 'v', 'beta', 'raw_g')]
        q, k, v, beta, g = inputs
        ref_inputs = [value.detach().float().requires_grad_() for value in inputs]
        ref, _ = reference_owner.recurrent_gated_delta_rule_ref(
            q=ref_inputs[0], k=ref_inputs[1], v=ref_inputs[2], beta=ref_inputs[3],
            g=ref_inputs[4], scale=scale, initial_state=initial, output_final_state=False)
        ref_grads = torch.autograd.grad(ref, ref_inputs, native_output_seed)
        captured = {}

        def capture(frame, event, value):
            if frame.f_code is stage.__code__ and event == 'return' and value is not None:
                for name in ('q', 'k', 'v', 'g', 'beta', 'A', 'w', 'v_new', 'h'):
                    captured[name] = frame.f_locals[name].detach().repeat_interleave(2, 0)

        assert sys.getprofile() is None
        sys.setprofile(capture)
        try:
            out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, beta=beta, g=g, scale=scale,
                initial_state=initial, output_final_state=False, use_qk_l2norm_in_kernel=False)
        finally:
            sys.setprofile(None)
        captured['raw_g'] = g.detach().repeat_interleave(2, 0)
        shift = exponent if dtype == torch.float16 else torch.zeros_like(exponent)
        do = torch.ldexp(native_output_seed, -shift).to(dtype)
        raw_native = torch.autograd.grad(out, inputs, do)
        native = [torch.ldexp(value.float(), shift if value.ndim == 4 else shift[..., 0])
                  for value in raw_native]
        with torch.no_grad():
            raw_finite = finite_owner(captured, do, scale)
            finite = {name: torch.ldexp(value.float(), shift if value.ndim == 4 else shift[..., 0])
                      for name, value in raw_finite.items()}
        checks = [check('native o', 'o', ref, out)]
        for name, key, expected, actual in zip(('q', 'k', 'v', 'beta', 'g'),
                ('dq', 'dk', 'dv', 'db', 'dg'), ref_grads, native):
            checks.append(check('range native d'+name, key, expected, actual))
            checks.append(check('range finite d'+name, key, expected, finite[name]))
        torch.cuda.synchronize()
        result['cases'].append(dict(dtype=str(dtype), native_seed_dtype=str(do.dtype),
            seed_maxabs=float(do.abs().max()), native_seed_nonfinite=int((~torch.isfinite(do)).sum()),
            input_dtypes=[str(value.dtype) for value in inputs], initial_state_nonzero=bool(initial.count_nonzero()),
            checks=checks, seconds=time.perf_counter()-tick,
            status='passed' if all(value['status']=='passed' for value in checks) else 'failed'))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['cases'][-1]), flush=True)
        del inputs, q, k, v, beta, g, ref_inputs, ref, ref_grads, out, captured
        del raw_native, native, raw_finite, finite, do
    result.update(status='completed_real_overflow_block_checks',
                  peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
