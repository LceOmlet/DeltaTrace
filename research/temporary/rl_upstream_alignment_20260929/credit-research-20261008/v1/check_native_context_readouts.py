"""Test original FLA readout composition on the saved real overflow block.

This is an operator/API test, not a DT estimator or a faithfulness test. The
original chunk readout, reverse-state stages, FP32 recurrence, and assert_close
own all numerical calculations and thresholds. No model forward is executed.
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
from fla.ops.common.chunk_o import chunk_fwd_o
from finite_fla_gpu import native_input_adjoints


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('operands', 'precast', 'official', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert not fla.utils.FLA_CI_ENV
    assert sha(args.official) == '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    owner = importlib.import_module('fla.ops.common.chunk_o')
    assert sha(owner.__file__) == '548cd026c316d61d74e691c82c82165b7d12bc437f47ec2ac1e1f50bb48396be'
    spec = importlib.util.spec_from_file_location('original_context_test', args.official)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    tree = ast.parse(args.official.read_bytes())
    test = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_chunk')
    thresholds = {n.args[0].value: n.args[3].value for n in ast.walk(test)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'assert_close'}
    saved = torch.load(args.operands, weights_only=False, mmap=True, map_location='cpu')
    precast = torch.load(args.precast, weights_only=False, mmap=True, map_location='cpu')
    endpoints, bad_seed, scale = saved['args']
    before = precast['mo'][:, :, 8:16].to(precast['output_dtype'])
    assert torch.equal(before.to(bad_seed.dtype), bad_seed)
    bad = (~torch.isfinite(bad_seed)).nonzero()
    assert len(bad) == 1
    first = int(bad[0, 1])//64*64
    last = min(first+128, before.shape[1])
    assert (last-first) % 64 == 0
    chunks = (last-first)//64
    original = {key: endpoints[key][1::2, first:last].contiguous()
                for key in ('q', 'k', 'v', 'beta', 'raw_g')}
    initial = endpoints['h'][1::2, first//64].float().cuda()
    original_seed = before[:, first:last].float().cuda()
    result = dict(scope=__doc__, script=dict(path=__file__, sha256=sha(__file__)),
        original_test=dict(path=str(args.official), sha256=sha(args.official), thresholds=thresholds),
        original_readout=dict(path=owner.__file__, sha256=sha(owner.__file__)),
        operands=dict(path=str(args.operands), sha256=sha(args.operands)),
        precast=dict(path=str(args.precast), sha256=sha(args.precast)),
        block=[first, last], cases=[], model_calls=0, DT_calls=0, optimizer=0,
        production_modified=False, attribution_accuracy_repaired=False)

    def check(name, key, expected, actual):
        item = dict(name=name, official_quantity=key, threshold=thresholds[key],
                    normalized_error=float(fla.utils.get_err_ratio(expected, actual)),
                    max_abs=float(fla.utils.get_abs_err(expected, actual)),
                    nonfinite=int((~torch.isfinite(actual)).sum()))
        try:
            fla.utils.assert_close(name, expected, actual, thresholds[key])
            item['status'] = 'passed'
        except AssertionError as error:
            item.update(status='failed', error=str(error))
        return item

    stage = importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd
    for dtype in (torch.float16, torch.bfloat16):
        tick = time.perf_counter()
        q, k, v, beta, raw_g = [original[key].cuda().to(torch.float32 if key == 'raw_g' else dtype)
                              for key in ('q', 'k', 'v', 'beta', 'raw_g')]
        # Use the existing minimum range representation; reference sees the
        # exact represented seed. No output credit enters this test.
        exponent = torch.ceil(torch.log2(original_seed.abs().amax((1, 3), keepdim=True)
                                         /torch.finfo(dtype).max).clamp_min(0)).int()
        do = torch.ldexp(original_seed, -exponent).to(dtype)
        ref_inputs = [value.float().detach().requires_grad_() for value in (q, k, v, beta, raw_g)]
        ref_out, _ = reference.recurrent_gated_delta_rule_ref(q=ref_inputs[0], k=ref_inputs[1],
            v=ref_inputs[2], beta=ref_inputs[3], g=ref_inputs[4], scale=scale,
            initial_state=initial, output_final_state=False)
        ref_grads = torch.autograd.grad(ref_out, ref_inputs, do.float())
        captured = {}

        def capture(frame, event, value):
            if frame.f_code is stage.__code__ and event == 'return' and value is not None:
                for name in ('q', 'k', 'v', 'g', 'beta', 'A', 'w', 'v_new', 'h'):
                    captured[name] = frame.f_locals[name].detach()

        assert sys.getprofile() is None
        sys.setprofile(capture)
        try:
            with torch.no_grad():
                native_out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, beta=beta, g=raw_g,
                    scale=scale, initial_state=initial, output_final_state=False,
                    use_qk_l2norm_in_kernel=False)
        finally:
            sys.setprofile(None)
        captured['raw_g'] = raw_g
        with torch.no_grad():
            # H_t do_t is exactly the native dq quantity. Reuse the original
            # readout on contiguous transposed rank-update factors and state.
            state_query = chunk_fwd_o(q=do.contiguous(), k=captured['v_new'].contiguous(),
                v=k.contiguous(), h=captured['h'].transpose(-1, -2).contiguous(),
                g=captured['g'], scale=scale)
            # The native output for q'=shift(q) reads the factual preceding
            # state. This is an unchanged original output operator and uses
            # its o threshold; no new tolerance is assigned to a renamed sum.
            shifted_q = torch.cat((q[:, 1:], torch.zeros_like(q[:, :1])), dim=1).contiguous()
            previous_raw_first = chunk_fwd_o(q=shifted_q, k=k, v=captured['v_new'],
                h=captured['h'], g=captured['g'], scale=scale)
            previous = chunk_fwd_o(q=shifted_q, k=k.contiguous(), v=captured['v_new'].contiguous(),
                h=captured['h'].contiguous(), g=captured['g'].contiguous(), scale=scale)
            replay = chunk_fwd_o(q=q.contiguous(), k=k.contiguous(),
                v=captured['v_new'].contiguous(), h=captured['h'].contiguous(),
                g=captured['g'].contiguous(), scale=scale)
            previous_raw_after = chunk_fwd_o(q=shifted_q, k=k, v=captured['v_new'],
                h=captured['h'], g=captured['g'], scale=scale)
            previous_captured_k = chunk_fwd_o(q=shifted_q, k=captured['k'], v=captured['v_new'],
                h=captured['h'], g=captured['g'], scale=scale)
            ref_previous, _ = reference.recurrent_gated_delta_rule_ref(q=shifted_q.float(),
                k=k.float(), v=v.float(), beta=beta.float(), g=raw_g,
                scale=scale, initial_state=initial, output_final_state=False)
            paired = {name: value.repeat_interleave(2, 0) for name, value in captured.items()}
            adjoints = native_input_adjoints(paired, do, scale)
            # Reuse the native WY coefficient transformation, not a second
            # reverse recurrence. Original 64-token chunk layout is explicit.
            def pack(value):
                return value.permute(0, 2, 1, 3).reshape(-1, 64, value.shape[-1]).contiguous()

            def unpack(value):
                return value.reshape(q.shape[0], q.shape[2], chunks*64, value.shape[-1]).permute(0, 2, 1, 3).contiguous()

            def reverse(value):
                shape = value.shape
                return value.reshape(shape[0], chunks, 64, *shape[2:]).flip(2).reshape(shape).contiguous()

            A = pack(captured['A'])
            du = pack(adjoints['dU_WY'])
            L = torch.bmm(A.transpose(-1, -2), du.to(dtype), out_dtype=torch.float32)
            L = unpack(L)
            W = (beta.float()[..., None]*L).to(dtype)
            G = captured['g'].reshape(q.shape[0], chunks, 64, q.shape[2])
            reverse_g = (G[:, :, -1:]-G).flip(2).reshape(q.shape[:3]).contiguous()
            # P_j^T k_j = boundary + inclusive(q do) - strict(k W).
            # Restore the omitted strict/inclusive diagonal term. The original
            # readout scales its boundary too, so divide its FP32 boundary by
            # the fixed positive model attention scale before that owner call.
            terminal = (adjoints['dh_end']/scale).contiguous()
            zero_terminal = torch.zeros_like(terminal)
            future_output = reverse(chunk_fwd_o(q=reverse(k), k=reverse(q),
                v=reverse(do), h=terminal, g=reverse_g, scale=scale))
            future_update = reverse(chunk_fwd_o(q=reverse(k), k=reverse(k),
                v=reverse(W), h=zero_terminal, g=reverse_g, scale=1.))
            L_query = future_output.float()-future_update.float()+(k.float()*k.float()).sum(-1, keepdim=True)*W.float()
            future_dv = beta.float()[..., None]*L_query
        checks = [check('native o', 'o', ref_out, native_out),
                  check('original readout replay o', 'o', ref_out, replay),
                  check('raw shifted-query first o', 'o', ref_previous, previous_raw_first),
                  check('native shifted-query o', 'o', ref_previous, previous),
                  check('raw shifted-query after replay o', 'o', ref_previous, previous_raw_after),
                  check('raw captured-key shifted-query o', 'o', ref_previous, previous_captured_k),
                  check('transposed state readout dq', 'dq', ref_grads[0], state_query),
                  check('reversed readout dv', 'dv', ref_grads[2], future_dv)]
        torch.cuda.synchronize()
        artifact = args.output.parent/('operands-'+str(dtype).split('.')[-1]+'.pt')
        torch.save(dict(captured={key:value.cpu() for key,value in captured.items()},
            external={key:value.cpu() for key,value in [('q',q),('k',k),('v',v),('beta',beta),('raw_g',raw_g)]},
            initial=initial.cpu(), do=do.cpu(), scale=scale,
            shifted_q=shifted_q.cpu(), native_out=native_out.cpu(), reference_out=ref_out.cpu(),
            reference_shifted=ref_previous.cpu(), raw_first=previous_raw_first.cpu(),
            contiguous=previous.cpu(), raw_after_replay=previous_raw_after.cpu(),
            raw_captured_k=previous_captured_k.cpu(),
            transposed_dq=state_query.cpu(), reference_dq=ref_grads[0].cpu(),
            reversed_dv=future_dv.cpu(), reference_dv=ref_grads[2].cpu()),artifact)
        result['cases'].append(dict(dtype=str(dtype), shape=list(q.shape),
            initial_state_nonzero=bool(initial.count_nonzero()),
            reverse_boundary_nonzero=bool(adjoints['dh_end'].count_nonzero()),
            actual_layouts={key:dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype))
                            for key,value in captured.items()},
            external_layouts={key:dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
                contiguous=value.is_contiguous(),contiguous_same_ptr=value.data_ptr()==value.contiguous().data_ptr(),
                captured_same_ptr=value.data_ptr()==captured[key].data_ptr())
                for key,value in [('q',q),('k',k),('v',v),('beta',beta),('raw_g',raw_g)]},
            unchanged_inputs={key:bool(torch.equal(value, original[key].cuda().to(value.dtype)))
                              for key,value in [('q',q),('k',k),('v',v),('beta',beta),('raw_g',raw_g)]},
            captured_query_matches_input=bool(torch.equal(q,captured['q'])),
            contiguous_identity={key:bool(value.data_ptr()==value.contiguous().data_ptr())
                                 for key,value in captured.items()},
            exact_artifact=dict(path=str(artifact), sha256=sha(artifact), bytes=artifact.stat().st_size),
            seed_exponents=exponent.cpu().flatten().tolist(), checks=checks,
            seconds=time.perf_counter()-tick,
            status='passed' if all(item['status']=='passed' for item in checks) else 'failed'))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='complete_operator_API_diagnostic',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  limits='Original o/dq/dv quantities only. No claim of arbitrary-query, full conditional-window coefficients, whole-method faithfulness, or training speed.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
