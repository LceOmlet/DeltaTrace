"""Check the new finite-memory composition at its original derivative limit.

Use the already preserved real B4/199-token original reference artifacts.
No new model/native forward is performed. The original FP32 reference fills
only the missing dg artifact. Finite effects and whole-method attribution
quality remain separate, unaccepted checks.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import fla.utils
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
    assert len(original['cases']) == 2
    assert all(c['status'] == 'passed' for case in original['cases'] for c in case['checks'])
    assert not fla.utils.FLA_CI_ENV
    torch.set_num_threads(8)
    thresholds = original['original_test']['thresholds']
    reference_path = original['original_test']['path']
    assert sha(reference_path) == original['original_test']['sha256']
    spec = importlib.util.spec_from_file_location('original_limit_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    result = dict(scope=__doc__, cases=[], original=dict(path=str(args.original/'result.json'),
        sha256=sha(args.original/'result.json')), original_test=original['original_test'],
        original_readout=original['original_readout'],
        sources={name:dict(path=importlib.import_module(name).__file__,
                          sha256=sha(importlib.import_module(name).__file__))
                 for name in ('finite_fla_gpu', 'native_conditional_queries', 'conditional_window_memory')},
        script=dict(path=__file__, sha256=sha(__file__)), model_calls=0, DT_calls=0,
        native_forward=0, reference_autograd=2, optimizer=0, production_modified=False,
        credit_repair_accepted=False)
    for old in original['cases']:
        tick = time.perf_counter()
        item = old['exact_artifact']
        assert sha(item['path']) == item['sha256']
        saved = torch.load(item['path'], mmap=True, map_location='cpu', weights_only=False)
        factual = {key:value.cuda().contiguous() for key,value in saved['capture'].items()}
        do = saved['do'].cuda().contiguous()
        B, T, H, K = factual['q'].shape
        assert T == 199
        N = (T+63)//64
        raw_g_ref = factual['raw_g'].float().detach().requires_grad_()
        reference_o, _ = reference.recurrent_gated_delta_rule_ref(
            q=factual['q'].float(), k=factual['k'].float(), v=factual['v'].float(),
            beta=factual['beta'].float(), g=raw_g_ref, scale=saved['scale'],
            initial_state=saved['initial'].cuda(), output_final_state=False)
        dg_reference, = torch.autograd.grad(reference_o, raw_g_ref, do.float())

        def unpack(value):
            return value.reshape(B, H, N*64, K)[:, :, :T].permute(0, 2, 1, 3).contiguous()

        def ahead(value, lag):
            return F.pad(value[:, lag:], (0, 0, 0, 0, 0, lag))

        with torch.no_grad():
            paired = {key:value.repeat_interleave(2, 0) for key,value in factual.items()}
            adjoints = native_input_adjoints(paired, do, saved['scale'])
            base, detail = mixed_coefficients(paired, adjoints, saved['scale'], diagnostics=True)
            L, r0 = unpack(detail['L']), unpack(detail['r0'])
            queries = NativeStateQueries(factual, adjoints, L, saved['scale'])
            conditional = {key:torch.stack([ahead(factual[key], lag) for lag in range(4)])
                           for key in ('q', 'k', 'v')}
            coefficient = conditional_memory_coefficients(factual, base, L, r0, queries,
                conditional, factual['raw_g'].float().exp(), factual['beta'], saved['scale'])
        checks = []
        expected = saved['reference']
        for name, key in (('q', 'dq'), ('k', 'dk'), ('v', 'dv')):
            for lag in range(4):
                wanted = ahead(expected[key].cuda(), lag)
                actual = coefficient[name][lag]
                row = dict(official_quantity=key, lag=lag, threshold=thresholds[key],
                    normalized_error=float(fla.utils.get_err_ratio(wanted, actual)),
                    max_abs=float(fla.utils.get_abs_err(wanted, actual)),
                    nonfinite=int((~torch.isfinite(actual)).sum()))
                try:
                    fla.utils.assert_close(key, wanted, actual, thresholds[key])
                    row['status'] = 'passed'
                except AssertionError as error:
                    row.update(status='failed', error=str(error))
                checks.append(row)
        for key, actual in (('db', coefficient['beta']),
                            ('dg', coefficient['alpha']*factual['raw_g'].float().exp())):
            # dg is the original derivative of exp(raw_g), not a tolerance
            # invented for the finite alpha scalar.
            wanted = dg_reference if key == 'dg' else expected[key].cuda()
            row = dict(official_quantity=key, threshold=thresholds[key],
                normalized_error=float(fla.utils.get_err_ratio(wanted, actual)),
                max_abs=float(fla.utils.get_abs_err(wanted, actual)),
                nonfinite=int((~torch.isfinite(actual)).sum()))
            try:
                fla.utils.assert_close(key, wanted, actual, thresholds[key])
                row['status'] = 'passed'
            except AssertionError as error:
                row.update(status='failed', error=str(error))
            checks.append(row)
        torch.cuda.synchronize()
        artifact = args.output.parent/('finite-limit-'+old['dtype'].split('.')[-1]+'.pt')
        torch.save(dict(coefficients={key:value.cpu() for key,value in coefficient.items()},
                        base={key:value.cpu() for key,value in base.items()},
                        original_FP32_dg=dg_reference.cpu()), artifact)
        result['cases'].append(dict(dtype=old['dtype'], shape=[B, T, H, K],
            checks=checks, original_readout_calls=queries.readout_calls,
            alpha_minus_coincident_max_abs=float((coefficient['alpha']-base['alpha']).abs().max()),
            alpha_comparison='Existing native coincident finite owner, supplementary to the original FP32 dg assertion.',
            artifact=dict(path=str(artifact), bytes=artifact.stat().st_size, sha256=sha(artifact)),
            seconds=time.perf_counter()-tick))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='complete_conditional_memory_derivative_limit_diagnostic',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        limits='Original dq/dk/dv/db/dg quantities, 28 checks with original tolerances. The missing dg uses the original FP32 reference on exact preserved operands. No nonzero finite-effect or full-model quality/efficiency acceptance.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
