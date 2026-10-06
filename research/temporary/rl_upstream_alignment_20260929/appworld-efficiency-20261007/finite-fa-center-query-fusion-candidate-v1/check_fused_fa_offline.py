"""Prepared-only, actual saved-operand FA fusion diagnostic; no model/CP input.

The unchanged 7ff checker owns every reference and numerical assertion. This
harness only transports already-saved full logical B1 rows and aliases one
diagnostic CDLL instance to the independently named fused row ABI. Noncoincident
fusion differences have no invented tolerance or whole-DT acceptance gate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time
import traceback


HELPER_SHA = 'e32fc0098c5cccb0cb8a5e87150484e407be35c279f5035eb52958c068c6dad0'
CANDIDATE_SHA = '7a6dfeb3d2f23903c8143d29a79e1e55b54effe1346221660924db5218f219ed'
ROW_ABI = 'deltatrace_fa_finite_p1_bf16_d256_row_cached_suffix'
FUSED_ABI = ROW_ABI + '_fused'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reuse-helper', 'source-preparation', 'candidate-source',
                 'native-operands', 'finite-operands', 'saved-verifier', 'sources',
                 'row-wrapper', 'baseline-library', 'candidate-library',
                 'environment-json', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-library-sha256', required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--complete-row', type=Path, action='append', required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new isolated output directory; existing receipts are preserved.')
    args.output_dir.mkdir(parents=True)
    report = dict(status='in_progress', rank=args.rank, official_rows=[],
                  model_loads=0, checkpoint_operations=0, random_operands_generated=False,
                  official_source_or_assertions_changed=False, formal_changes=False,
                  source_checks={}, selected_fused_ABI=FUSED_ABI,
                  kernel_launches=dict(original_default=3, fused_opt_in=2),
                  limits=['Original7ff coincident derivative-limit assertions only.',
                          'Noncoincident five-output differences are diagnostics, not a finite tolerance.',
                          'Timing includes unchanged Python wrapper, allocation, conversion and synchronization.'])
    output = args.output_dir / 'fusion-offline-result.json'

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')

    torch = None
    try:
        # Reuse original transport utilities; no original main/test is invoked here.
        if hashlib.sha256(args.reuse_helper.read_bytes()).hexdigest() != HELPER_SHA:
            raise ValueError('Original actual-row transport helper changed.')
        spec = importlib.util.spec_from_file_location('_original_actual_row_transport', args.reuse_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        sources = report['source_checks']
        for key, path, expected in (
            ('original_transport_helper', args.reuse_helper, HELPER_SHA),
            ('original_verifier', args.saved_verifier, helper.VERIFIER_SHA),
            ('original_load_helper', args.saved_verifier.parent / 'verify_official_kernel_tolerances.py', helper.LOAD_HELPER_SHA),
            ('original_FA_tests', args.sources / 'test_flash_attn_v263.py', helper.FA_SOURCE_SHA),
            ('unchanged_row_wrapper', args.row_wrapper, helper.ROW_WRAPPER_SHA),
            ('baseline_library', args.baseline_library, helper.ROW_LIBRARY_SHA),
            ('candidate_library', args.candidate_library, args.candidate_library_sha256),
            ('candidate_source', args.candidate_source, CANDIDATE_SHA),
        ):
            sources[key] = helper.source_check(path, expected)
        for key, path in (('source_preparation', args.source_preparation),
                          ('environment', args.environment_json), ('harness', Path(__file__))):
            sources[key] = dict(path=str(path), sha256=helper.sha(path))
        prepared = json.loads(args.source_preparation.read_bytes())
        if prepared['candidate']['sha256'] != CANDIDATE_SHA:
            raise ValueError('Prepared metadata does not bind this CUDA candidate.')
        assets = next(row for row in prepared['actual_saved_assets'] if row['rank'] == args.rank)
        sources['native_operands'] = helper.source_check(args.native_operands, assets['native_operands']['sha256'])
        sources['finite_operands'] = helper.source_check(args.finite_operands, assets['finite_operands']['sha256'])
        rows = assets['existing_logical_B1_rows']
        if len(rows) != 4 or len(args.complete_row) != len(rows):
            raise ValueError('Require exactly the four existing complete logical B1 rows in row order.')
        report['complete_row_sources'] = [helper.source_check(path, row['operands']['sha256'])
                                         for path, row in zip(args.complete_row, rows)]
        save()

        import torch as torch_module
        torch = torch_module
        wrapper = helper.module('_unmodified_fusion_diagnostic_wrapper', args.row_wrapper)
        original_layout, original_factory = wrapper.RightPaddedLengths, wrapper.VendorFAFiniteP1BF16D256
        baseline = original_factory(args.baseline_library, helper.ROW_LIBRARY_SHA)
        candidate_default = original_factory(args.candidate_library, args.candidate_library_sha256)
        fused = original_factory(args.candidate_library, args.candidate_library_sha256)
        # CDLL attributes are per instance; the default owners above retain their ABI.
        setattr(fused.library, ROW_ABI, getattr(fused.library, FUSED_ABI))

        class ActualRowLayout(original_layout):
            """Same scalar-B1 to row-layout transport as original e32 helper."""
            def __init__(self, valid_lengths, padded_length, device, **kwargs):
                start = kwargs.pop('query_start', 0)
                coefficients = kwargs.pop('coefficient_starts', None)
                super().__init__(valid_lengths, padded_length, device,
                    coefficient_starts=coefficients, query_starts=[start] * len(valid_lengths),
                    query_padded_length=padded_length - start, **kwargs)

        previous_module = sys.modules.get('vendor_fa_finite_bf16_d256')
        previous_argv, previous_path = sys.argv, list(sys.path)
        previous_env = os.environ.get('DT_ENVIRONMENT_JSON')
        try:
            wrapper.RightPaddedLengths = ActualRowLayout
            wrapper.VendorFAFiniteP1BF16D256 = lambda _library, _sha: fused
            sys.modules['vendor_fa_finite_bf16_d256'] = wrapper
            sys.path.insert(0, str(args.saved_verifier.parent))
            os.environ['DT_ENVIRONMENT_JSON'] = str(args.environment_json)
            for index, (path, row) in enumerate(zip(args.complete_row, rows)):
                result_path = args.output_dir / f'original-seven-assertions-row{index}.json'
                sys.argv = [str(args.saved_verifier), '--operands', str(path),
                            '--sources', str(args.sources), '--output', str(result_path)]
                print(f'phase=original_unchanged_seven_assertions row={index}', flush=True)
                runpy.run_path(str(args.saved_verifier), run_name='__main__')
                result = json.loads(result_path.read_bytes())
                if len(result['checks']) != 7:
                    raise ValueError('Original verifier did not return its seven checks.')
                report['official_rows'].append(dict(row=index, Q=row['Q'], K=row['K'],
                    query_start=row['query_start'], coefficient_start=row['coefficient_start'],
                    original_result=dict(path=str(result_path), sha256=helper.sha(result_path), status=result['status'])))
                save()
                if result['status'] != 'passed':
                    raise RuntimeError(f'Original unchanged FA assertions failed on logical row {index}; result preserved.')
                gc.collect()
        finally:
            wrapper.RightPaddedLengths, wrapper.VendorFAFiniteP1BF16D256 = original_layout, original_factory
            sys.argv, sys.path = previous_argv, previous_path
            if previous_module is None:
                sys.modules.pop('vendor_fa_finite_bf16_d256', None)
            else:
                sys.modules['vendor_fa_finite_bf16_d256'] = previous_module
            if previous_env is None:
                os.environ.pop('DT_ENVIRONMENT_JSON', None)
            else:
                os.environ['DT_ENVIRONMENT_JSON'] = previous_env

        saved = torch.load(args.finite_operands, map_location='cpu', weights_only=True, mmap=True)
        if saved['decoder_index'] != 3 or saved['invocation'] != 8 or set(saved['operands']) != set(helper.OPERANDS):
            raise ValueError('Not the existing actual eighth finite FA call at decoder3.')
        ops = {name: value.contiguous().to('cuda') for name, value in saved['operands'].items()}
        layout = original_layout(list(saved['lengths']), ops['k0'].shape[2], 'cuda',
            coefficient_starts=list(saved['coefficient_starts']), query_starts=list(saved['query_starts']),
            query_padded_length=ops['q0'].shape[2])
        report['actual_B4_layout'] = dict(lengths=list(saved['lengths']), query_starts=list(saved['query_starts']),
            coefficient_starts=list(saved['coefficient_starts']), query_padded_length=ops['q0'].shape[2],
            operand_dtypes={name: str(value.dtype) for name, value in ops.items()})

        def difference(actual, expected):
            checks = {}
            for name in helper.OUTPUTS:
                a, b = actual[name].detach().cpu(), expected[name].detach().cpu()
                delta = a.double() - b.double()
                checks[name] = dict(equal=torch.equal(a, b), max_abs=float(delta.abs().max()),
                    L2=float(torch.linalg.vector_norm(delta)), finite=bool(torch.isfinite(a).all()),
                    shape=list(a.shape), dtype=str(a.dtype))
            return checks

        with torch.no_grad():
            print('phase=actual_nonzero_B4_default_regression_and_fused_diagnostic', flush=True)
            original = baseline(ops, saved['scale'], layout)
            default = candidate_default(ops, saved['scale'], layout)
            selected = fused(ops, saved['scale'], layout)
            torch.cuda.synchronize()
            report['candidate_default_vs_original_4f42'] = difference(default, original)
            report['candidate_fused_vs_original_4f42_noncoincident_diagnostic'] = difference(selected, original)
            del original, default, selected
            save()
            report['same_input_timings'] = {}
            for label, owner in (('original_4f42', baseline), ('candidate_fused', fused)):
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    value = owner(ops, saved['scale'], layout)
                    torch.cuda.synchronize()
                    del value
                seconds = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    value = owner(ops, saved['scale'], layout)
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - started)
                    del value
                report['same_input_timings'][label] = dict(warmup_calls=2, measured_calls=5,
                    seconds=seconds, median_seconds=statistics.median(seconds),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                save()
        passed = all(row['original_result']['status'] == 'passed' for row in report['official_rows'])
        unchanged = all(item['equal'] for item in report['candidate_default_vs_original_4f42'].values())
        report['official_status'] = 'passed' if passed else 'failed'
        report['default_ABI_regression_status'] = 'passed' if unchanged else 'failed'
        report['noncoincident_fused_outputs_finite'] = all(item['finite'] for item in
            report['candidate_fused_vs_original_4f42_noncoincident_diagnostic'].values())
        report['noncoincident_fusion_tolerance_status'] = 'not_defined_diagnostic_only'
        report['status'] = ('completed' if passed and unchanged and report['noncoincident_fused_outputs_finite']
                            else 'failed_original_assertion_default_ABI_regression_or_nonfinite_output')
    except Exception:
        report['status'] = 'failed'
        report['failure_traceback'] = traceback.format_exc()
    finally:
        if torch is not None and torch.cuda.is_initialized():
            report['torch_peak_allocated_bytes_at_exit'] = torch.cuda.max_memory_allocated()
            report['torch_peak_reserved_bytes_at_exit'] = torch.cuda.max_memory_reserved()
        save()
    print(json.dumps(dict(output=str(output), status=report['status']), indent=2), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
