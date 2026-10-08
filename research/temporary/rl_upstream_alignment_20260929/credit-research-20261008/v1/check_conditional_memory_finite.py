"""Nonzero finite-window identity on every source in the saved real span.

The original FP32 recurrent reference and native FLA own all factual and
conditional forward values. This diagnoses the local memory composition; it
does not replace frozen full-model cumulative deletion/RISE/MAS evaluation.
No official tolerance is invented for the contracted finite scalar.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from finite_fla_gpu import native_input_adjoints, mixed_coefficients
from native_conditional_queries import NativeStateQueries
from conditional_window_memory import conditional_memory_coefficients


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.original/'result.json').read_bytes())
    assert sha(original['original_test']['path']) == original['original_test']['sha256']
    assert not fla.utils.FLA_CI_ENV
    spec = importlib.util.spec_from_file_location('original_finite_reference', original['original_test']['path'])
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    raw_path = original['operands']['path']
    assert sha(raw_path) == original['operands']['sha256']
    raw = torch.load(raw_path, map_location='cpu', mmap=True, weights_only=False)['args'][0]
    first, last = original['block']
    torch.set_num_threads(8)
    result = dict(scope=__doc__, cases=[], script=dict(path=__file__, sha256=sha(__file__)),
        original_test=original['original_test'], original_result=dict(path=str(args.original/'result.json'),
        sha256=sha(args.original/'result.json')), original_operands=original['operands'],
        model_calls=0, DT_calls=0, optimizer=0, production_modified=False,
        credit_repair_accepted=False, finite_scalar_official_tolerance=None)
    with torch.no_grad():
        for old in original['cases']:
            tick = time.perf_counter()
            assert sha(old['exact_artifact']['path']) == old['exact_artifact']['sha256']
            saved = torch.load(old['exact_artifact']['path'], mmap=True, map_location='cpu', weights_only=False)
            f = {key:value.cuda().contiguous() for key,value in saved['capture'].items()}
            B, T, H, K = f['q'].shape
            dtype, scale = f['q'].dtype, saved['scale']
            do, initial = saved['do'].cuda(), saved['initial'].cuda()
            c = {key:raw[key][0::2, first:last].cuda().to(f[key].dtype).contiguous()
                 for key in ('q', 'k', 'v', 'raw_g', 'beta')}

            def ahead(value, lag):
                return F.pad(value[:, lag:], (0, 0, 0, 0, 0, lag))

            def unpack(value):
                return value.reshape(B, H, ((T+63)//64)*64, K)[:, :, :T].permute(0, 2, 1, 3).contiguous()

            paired = {key:value.repeat_interleave(2, 0) for key,value in f.items()}
            adjoints = native_input_adjoints(paired, do, scale)
            base, detail = mixed_coefficients(paired, adjoints, scale, diagnostics=True)
            L, r0 = unpack(detail['L']), unpack(detail['r0'])
            queries = NativeStateQueries(f, adjoints, L, scale)
            conditional = {key:torch.stack([ahead(c[key], lag) for lag in range(4)])
                           for key in ('q', 'k', 'v')}
            coefficients = conditional_memory_coefficients(f, base, L, r0, queries,
                conditional, c['raw_g'].float().exp(), c['beta'], scale)
            attributed = sum((coefficients[key][lag]*(ahead(f[key], lag).float()
                                -conditional[key][lag].float())).sum((-1, -2))
                             for key in ('q', 'k', 'v') for lag in range(4))
            attributed += (coefficients['alpha']*(f['raw_g'].float().exp()
                             -c['raw_g'].float().exp())).sum(-1)
            attributed += (coefficients['beta']*(f['beta'].float()-c['beta'].float())).sum(-1)
            factual_J = (saved['reference']['o'].cuda().float()*do.float()).sum((1, 2, 3))
            native_factual_J = (saved['actual']['o'].cuda().float()*do.float()).sum((1, 2, 3))
            expected = torch.empty_like(attributed)
            native_delta = torch.empty_like(attributed)
            owner_checks = []
            # All 199 sources in order, no tail/example selection. The source
            # grouping only bounds this diagnostic reference's state storage.
            for start in range(0, T, 16):
                indices = list(range(start, min(start+16, T)))
                E = len(indices)
                values = {key:f[key].repeat(E, 1, 1, 1) if f[key].ndim == 4 else f[key].repeat(E, 1, 1)
                          for key in ('q', 'k', 'v', 'raw_g', 'beta')}
                for row, source in enumerate(indices):
                    stop = min(source+4, T)
                    for key in ('q', 'k', 'v'):
                        values[key][row*B:(row+1)*B, source:stop] = c[key][:, source:stop]
                    for key in ('raw_g', 'beta'):
                        values[key][row*B:(row+1)*B, source] = c[key][:, source]
                init = initial.repeat(E, 1, 1, 1)
                ref_o, _ = reference.recurrent_gated_delta_rule_ref(
                    q=values['q'].float(), k=values['k'].float(), v=values['v'].float(),
                    beta=values['beta'].float(), g=values['raw_g'].float(), scale=scale,
                    initial_state=init, output_final_state=False)
                native_o, _ = chunk_gated_delta_rule(q=values['q'], k=values['k'], v=values['v'],
                    beta=values['beta'], g=values['raw_g'], scale=scale, initial_state=init,
                    output_final_state=False, use_qk_l2norm_in_kernel=False)
                row = dict(source_range=[indices[0], indices[-1]+1],
                    normalized_error=float(fla.utils.get_err_ratio(ref_o, native_o)),
                    max_abs=float(fla.utils.get_abs_err(ref_o, native_o)),
                    threshold=original['original_test']['thresholds']['o'])
                try:
                    fla.utils.assert_close('o', ref_o, native_o, row['threshold'])
                    row['status'] = 'passed'
                except AssertionError as error:
                    row.update(status='failed', error=str(error))
                owner_checks.append(row)
                seed = do.repeat(E, 1, 1, 1).float()
                ref_J = (ref_o.float()*seed).sum((1, 2, 3)).reshape(E, B).T
                native_J = (native_o.float()*seed).sum((1, 2, 3)).reshape(E, B).T
                expected[:, indices] = factual_J[:, None]-ref_J
                native_delta[:, indices] = native_factual_J[:, None]-native_J
            artifact = args.output.parent/('nonzero-finite-'+old['dtype'].split('.')[-1]+'.pt')
            torch.save(dict(attributed=attributed.cpu(), original_FP32_delta=expected.cpu(),
                original_native_delta=native_delta.cpu(),
                conditional={key:value.cpu() for key,value in c.items()},
                coefficients={key:value.cpu() for key,value in coefficients.items()}), artifact)
            error = attributed-expected
            result['cases'].append(dict(dtype=old['dtype'], shape=[B,T,H,K],
                sources_per_trajectory=T, source_selection='every position, uniform census',
                nonzero_reference_effects=int(expected.count_nonzero()),
                finite_scalar=dict(normalized_RMS=float(fla.utils.get_err_ratio(expected, attributed)),
                    max_abs=float(error.abs().max()), nonfinite=int((~torch.isfinite(attributed)).sum()),
                    reference_RMS=float(expected.square().mean().sqrt()),
                    residual_RMS=float(error.square().mean().sqrt()),
                    official_pass_claim=False), original_output_checks=owner_checks,
                original_readout_calls=queries.readout_calls,
                artifact=dict(path=str(artifact), bytes=artifact.stat().st_size, sha256=sha(artifact)),
                seconds=time.perf_counter()-tick))
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='complete_nonzero_local_memory_identity_diagnostic',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        limits='Only local memory with a fixed supplied cotangent. Four-position q/k/v interventions use real saved endpoint values; they are not yet the selected hidden-row causal-conv integration. Scalar finite error has no original FLA tolerance and is reported without an official pass. No whole-method repair or speed claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
