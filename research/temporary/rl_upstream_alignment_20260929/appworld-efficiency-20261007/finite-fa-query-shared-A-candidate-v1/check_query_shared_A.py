"""Isolated same-ABI shared-A diagnostic on existing actual FA operands.

Reuse the unchanged e32 transport main and its unchanged 7ff FA assertions.
Noncoincident five-output comparisons are diagnostics without a new tolerance.
No model, checkpoint, training, profiler, or production source is involved.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


HELPER_SHA = 'e32fc0098c5cccb0cb8a5e87150484e407be35c279f5035eb52958c068c6dad0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--candidate-library', type=Path, required=True)
    parser.add_argument('--candidate-library-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; existing receipts are preserved.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_output = args.output.parent / (args.output.stem + '-original-rows')
    if row_output.exists():
        raise FileExistsError('Use a new output path with a new original-row directory.')
    report = dict(status='in_progress', rank=args.rank, source_checks={},
        model_loads=0, checkpoint_operations=0, formal_changes=False,
        random_operands_generated=False, official_source_or_assertions_changed=False,
        selected_ABI='deltatrace_fa_finite_p1_bf16_d256_row_cached_suffix',
        noncoincident_tolerance='not_defined_diagnostic_only',
        timing_scope='Unprofiled original wrapper, allocations, conversions and synchronization.',
        memory_scope='Torch allocator peaks and instantaneous device free/total; not physical peak VRAM.')

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')

    torch = None
    try:
        assets = json.loads(args.assets.read_bytes())
        helper_path = Path(assets['transport_helper']['path'])
        import hashlib
        if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
            raise ValueError('Original e32 actual-row transport helper changed.')
        spec = importlib.util.spec_from_file_location('_isolated_shared_A_actual_row_helper', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        paths = {}
        expected = dict(transport_helper=HELPER_SHA,
            deployed_wrapper=helper.ROW_WRAPPER_SHA, deployed_library=helper.ROW_LIBRARY_SHA,
            official_verifier=helper.VERIFIER_SHA,
            scalar_wrapper=helper.SCALAR_WRAPPER_SHA, scalar_library=helper.SCALAR_LIBRARY_SHA)
        for name, sha256 in expected.items():
            item = assets[name]
            if item['sha256'] != sha256:
                raise ValueError(f'Assets identify a different original {name}.')
            paths[name] = Path(item['path'])
            report['source_checks'][name] = helper.source_check(paths[name], sha256)
        item = assets['environment_json']
        paths['environment_json'] = Path(item['path'])
        report['source_checks']['environment_json'] = dict(
            path=str(paths['environment_json']), sha256=helper.sha(paths['environment_json']))
        paths['official_sources'] = Path(assets['official_sources']['path'])
        report['source_checks']['original_FA_tests'] = helper.source_check(
            paths['official_sources'] / 'test_flash_attn_v263.py', helper.FA_SOURCE_SHA)
        report['source_checks']['original_load_helper'] = helper.source_check(
            paths['official_verifier'].parent / 'verify_official_kernel_tolerances.py', helper.LOAD_HELPER_SHA)
        report['source_checks']['candidate_library'] = helper.source_check(
            args.candidate_library, args.candidate_library_sha256)
        report['source_checks']['assets'] = dict(path=str(args.assets), sha256=helper.sha(args.assets))
        report['source_checks']['driver'] = dict(path=__file__, sha256=helper.sha(__file__))
        rank_assets = next(row for row in assets['saved_operands'] if row['rank'] == args.rank)
        for name in ('native_operands', 'finite_operands'):
            item = rank_assets[name]
            paths[name] = Path(item['path'])
            report['source_checks'][name] = helper.source_check(paths[name], item['sha256'])
        save()

        # Only this isolated module's dependency identity changes. Its main,
        # original scalar owner, complete-row transport and assertions do not.
        previous_sha, previous_argv = helper.ROW_LIBRARY_SHA, sys.argv
        try:
            helper.ROW_LIBRARY_SHA = args.candidate_library_sha256
            sys.argv = [str(helper_path)]
            for name, value in (
                ('native-operands', paths['native_operands']),
                ('finite-operands', paths['finite_operands']),
                ('saved-verifier', paths['official_verifier']),
                ('sources', paths['official_sources']),
                ('row-wrapper', paths['deployed_wrapper']),
                ('row-library', args.candidate_library),
                ('scalar-wrapper', paths['scalar_wrapper']),
                ('scalar-library', paths['scalar_library']),
                ('environment-json', paths['environment_json']),
                ('output-dir', row_output)):
                sys.argv.extend(['--' + name, str(value)])
            print('phase=original_e32_main_four_complete_rows_seven_assertions_each', flush=True)
            helper.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise RuntimeError(f'Original e32 main failed with exit status {exc.code}.') from exc
        finally:
            helper.ROW_LIBRARY_SHA, sys.argv = previous_sha, previous_argv
        original_result = row_output / 'actual-row-official-and-nonzero.json'
        report['original_helper_result'] = dict(path=str(original_result),
            sha256=helper.sha(original_result), result=json.loads(original_result.read_bytes()))
        save()

        import torch as torch_module
        torch = torch_module
        gc.collect()
        wrapper = helper.module('_isolated_shared_A_unchanged_row_wrapper', paths['deployed_wrapper'])
        baseline = wrapper.VendorFAFiniteP1BF16D256(paths['deployed_library'], helper.ROW_LIBRARY_SHA)
        candidate = wrapper.VendorFAFiniteP1BF16D256(args.candidate_library, args.candidate_library_sha256)
        saved = torch.load(paths['finite_operands'], map_location='cpu', weights_only=True, mmap=True)
        if saved['decoder_index'] != 3 or saved['invocation'] != 8:
            raise ValueError('Not the existing actual eighth finite FA call at decoder3.')
        ops = {name: value.contiguous().to('cuda') for name, value in saved['operands'].items()}
        layout = wrapper.RightPaddedLengths(list(saved['lengths']), ops['k0'].shape[2], 'cuda',
            coefficient_starts=list(saved['coefficient_starts']),
            query_starts=list(saved['query_starts']), query_padded_length=ops['q0'].shape[2])
        report['actual_B4'] = dict(decoder_index=saved['decoder_index'], invocation=saved['invocation'],
            lengths=list(saved['lengths']), query_starts=list(saved['query_starts']),
            coefficient_starts=list(saved['coefficient_starts']),
            operands={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in ops.items()})
        with torch.no_grad():
            print('phase=actual_noncoincident_B4_five_output_diagnostic', flush=True)
            before, after = baseline(ops, saved['scale'], layout), candidate(ops, saved['scale'], layout)
            torch.cuda.synchronize()
            report['candidate_vs_original_4f42'] = {}
            for name in helper.OUTPUTS:
                a, b = after[name].detach().cpu(), before[name].detach().cpu()
                report['candidate_vs_original_4f42'][name] = dict(equal=torch.equal(a, b),
                    max_abs=float((a.double() - b.double()).abs().max()),
                    finite=bool(torch.isfinite(a).all()), shape=list(a.shape), dtype=str(a.dtype))
            del before, after, a, b
            save()
            report['same_input_timings'] = {}
            for label, owner in (('original_4f42', baseline), ('candidate_shared_A', candidate)):
                print(f'phase=unprofiled_two_warm_five_measured owner={label}', flush=True)
                torch.cuda.reset_peak_memory_stats()
                free_before, total = torch.cuda.mem_get_info()
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
                free_after, _ = torch.cuda.mem_get_info()
                report['same_input_timings'][label] = dict(warmup_calls=2, measured_calls=5,
                    seconds=seconds, median_seconds=statistics.median(seconds),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    instantaneous_device_free_before_bytes=free_before,
                    instantaneous_device_free_after_bytes=free_after, device_total_bytes=total)
                save()
        report['status'] = 'completed_original_checks_and_noncoincident_diagnostics'
    except Exception:
        report['status'] = 'failed'
        report['failure_traceback'] = traceback.format_exc()
    finally:
        if torch is not None and torch.cuda.is_initialized():
            report['torch_peak_allocated_bytes_at_exit'] = torch.cuda.max_memory_allocated()
            report['torch_peak_reserved_bytes_at_exit'] = torch.cuda.max_memory_reserved()
        save()
    print(json.dumps(dict(output=str(args.output), status=report['status'])), flush=True)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
