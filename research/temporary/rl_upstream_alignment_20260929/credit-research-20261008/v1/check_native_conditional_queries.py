"""Original dtype derivative checks for conditional-query owner composition.

Saved real B4/head8 inputs, nonzero original initial state, all positions in a
199-token span including chunk boundaries and a partial final chunk. This
checks original derivative quantities, not nonzero finite-credit quality.
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
from finite_fla_gpu import native_input_adjoints, mixed_coefficients
from native_conditional_queries import NativeStateQueries


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
    spec = importlib.util.spec_from_file_location('original_query_reference', args.official)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    test = next(n for n in ast.parse(args.official.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'test_chunk')
    thresholds = {n.args[0].value:n.args[3].value for n in ast.walk(test)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == 'assert_close'}
    saved = torch.load(args.operands, map_location='cpu', mmap=True, weights_only=False)
    precast = torch.load(args.precast, map_location='cpu', mmap=True, weights_only=False)
    endpoints, bad_seed, scale = saved['args']
    original_seed = precast['mo'][:, :, 8:16].to(precast['output_dtype'])
    assert torch.equal(original_seed.to(bad_seed.dtype), bad_seed)
    first = int((~torch.isfinite(bad_seed)).nonzero()[0, 1])//64*64
    last = first+199
    assert last < bad_seed.shape[1]
    original = {key:endpoints[key][1::2, first:last].contiguous()
                for key in ('q', 'k', 'v', 'beta', 'raw_g')}
    initial = endpoints['h'][1::2, first//64].float().cuda()
    raw_seed = original_seed[:, first:last].float().cuda()
    readout_owner = importlib.import_module('fla.ops.common.chunk_o')
    adapter = importlib.import_module('native_conditional_queries')
    assert sha(readout_owner.__file__) == '548cd026c316d61d74e691c82c82165b7d12bc437f47ec2ac1e1f50bb48396be'
    result = dict(scope=__doc__, block=[first, last], cases=[],
        script=dict(path=__file__, sha256=sha(__file__)),
        adapter=dict(path=adapter.__file__, sha256=sha(adapter.__file__)),
        original_readout=dict(path=readout_owner.__file__, sha256=sha(readout_owner.__file__)),
        original_test=dict(path=str(args.official), sha256=sha(args.official), thresholds=thresholds),
        operands=dict(path=str(args.operands), sha256=sha(args.operands)),
        precast=dict(path=str(args.precast), sha256=sha(args.precast)),
        model_calls=0, DT_calls=0, optimizer=0, production_modified=False,
        credit_repair_accepted=False)
    stage = importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd
    for dtype in (torch.float16, torch.bfloat16):
        tick = time.perf_counter()
        q, k, v, beta, raw_g = [original[key].cuda().to(torch.float32 if key == 'raw_g' else dtype).contiguous()
                              for key in ('q', 'k', 'v', 'beta', 'raw_g')]
        exponent = torch.ceil(torch.log2(raw_seed.abs().amax((1, 3), keepdim=True)
                                         /torch.finfo(dtype).max).clamp_min(0)).int()
        do = torch.ldexp(raw_seed, -exponent).to(dtype).contiguous()
        inputs = [value.float().detach().requires_grad_() for value in (q, k, v, beta, raw_g)]
        expected, _ = reference.recurrent_gated_delta_rule_ref(q=inputs[0], k=inputs[1],
            v=inputs[2], beta=inputs[3], g=inputs[4], scale=scale,
            initial_state=initial, output_final_state=False)
        gradients = torch.autograd.grad(expected, inputs, do.float())
        capture = {}

        def observe(frame, event, value):
            if frame.f_code is stage.__code__ and event == 'return' and value is not None:
                for name in ('q', 'k', 'v', 'g', 'beta', 'A', 'w', 'v_new', 'h'):
                    capture[name] = frame.f_locals[name].detach()

        assert sys.getprofile() is None
        sys.setprofile(observe)
        try:
            with torch.no_grad():
                actual, _ = chunk_gated_delta_rule(q=q, k=k, v=v, beta=beta, g=raw_g,
                    scale=scale, initial_state=initial, output_final_state=False,
                    use_qk_l2norm_in_kernel=False)
        finally:
            sys.setprofile(None)
        capture['raw_g'] = raw_g
        paired = {key:value.repeat_interleave(2, 0) for key, value in capture.items()}
        with torch.no_grad():
            adjoints = native_input_adjoints(paired, do, scale)
            _, detail = mixed_coefficients(paired, adjoints, scale, diagnostics=True)
            B, T, H, K = q.shape
            C = 64
            N = (T+C-1)//C
            # Representation-only unpack of the existing owner's L artifact.
            L = detail['L'].reshape(B, H, N*C, K)[:, :, :T].permute(0, 2, 1, 3).contiguous()
            context = NativeStateQueries(capture, adjoints, L, scale)
            alpha = raw_g.float().exp()[..., None]
            past_k = context.past(k)
            dq = scale*context.current(do, transpose=True).float()
            dk = context.future(capture['v_new'], transpose=True)
            dk -= alpha*context.past((beta.float()[..., None]*L).to(dtype), transpose=True)
            dv = beta.float()[..., None]*context.future(k)
            db = ((v.float()-alpha*past_k)*L).sum(-1)
        values = [('o', expected, actual), ('dq', gradients[0], dq),
                  ('dk', gradients[1], dk), ('dv', gradients[2], dv),
                  ('db', gradients[3], db)]
        checks = []
        for key, wanted, got in values:
            row = dict(official_quantity=key, threshold=thresholds[key],
                normalized_error=float(fla.utils.get_err_ratio(wanted, got)),
                max_abs=float(fla.utils.get_abs_err(wanted, got)),
                nonfinite=int((~torch.isfinite(got)).sum()))
            try:
                fla.utils.assert_close(key, wanted, got, thresholds[key])
                row['status'] = 'passed'
            except AssertionError as error:
                row.update(status='failed', error=str(error))
            checks.append(row)
        torch.cuda.synchronize()
        artifact = args.output.parent/('query-operands-'+str(dtype).split('.')[-1]+'.pt')
        torch.save(dict(capture={key:value.cpu() for key,value in capture.items()},
            initial=initial.cpu(), do=do.cpu(), scale=scale,
            L=L.cpu(), past_k=past_k.cpu(),
            actual={key:value.cpu() for key,_,value in values},
            reference={key:value.cpu() for key,value,_ in values}), artifact)
        result['cases'].append(dict(dtype=str(dtype), shape=list(q.shape),
            first_chunk_state_nonzero=bool(initial.count_nonzero()),
            future_chunk_state_nonzero=bool(adjoints['dh_end'].count_nonzero()),
            original_readout_calls=context.readout_calls, checks=checks,
            exact_artifact=dict(path=str(artifact), bytes=artifact.stat().st_size, sha256=sha(artifact)),
            seconds=time.perf_counter()-tick))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='complete_bounded_conditional_query_derivative_checks',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        limits='Original o/dq/dk/dv/db quantities and original thresholds only. This verifies past/future query composition, including partial native chunks. Conditional finite coefficients and whole-method attribution are not tested or accepted.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
