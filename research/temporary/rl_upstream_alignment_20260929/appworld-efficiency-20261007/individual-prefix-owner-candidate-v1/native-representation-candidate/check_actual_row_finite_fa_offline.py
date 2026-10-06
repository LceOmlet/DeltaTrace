"""Offline original saved-FA verifier DI on actual heterogeneous row captures.

No model or checkpoint is involved. The original 7ff verifier executes without
source changes on each complete logical B1 row. Nonzero finite outputs are an
exact representation comparison against the saved B4 row, not a new tolerance.
This file is prepared only until explicitly executed after diagnostic release.
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
import sys
import time


VERIFIER_SHA = '7ff11d9d800a2c233b019213ae0fa9a5b603dfea51dfffabe6b9296a74aab9a9'
LOAD_HELPER_SHA = 'e9ee6be0bb59bdddb99400c7a0c14af775b75fa98b0bcbf744781f26f9b2f2a8'
FA_SOURCE_SHA = 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
ROW_WRAPPER_SHA = '3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'
ROW_LIBRARY_SHA = '4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'
SCALAR_WRAPPER_SHA = 'f5ea2f67af2a47b2a736545b22ff6683336be14be505060674fce6f01de3f15c'
SCALAR_LIBRARY_SHA = '5d2af760abb2684401682029e41ac68aefc58ed2796082d435874a2b74bbcebb'
OPERANDS = ('q0', 'q1', 'k0', 'k1', 'v0', 'u', 'lse0', 'lse1')
OUTPUTS = ('dq', 'dk', 'dv', 'tau', 'center')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    spec.loader.exec_module(owner)
    return owner


def source_check(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Original/candidate owner source hash mismatch: {path}: {actual}')
    return dict(path=str(path), sha256=actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-operands', 'finite-operands', 'saved-verifier', 'sources',
                 'row-wrapper', 'row-library', 'scalar-wrapper', 'scalar-library',
                 'environment-json', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sources = dict(
        original_verifier=source_check(args.saved_verifier, VERIFIER_SHA),
        original_load_helper=source_check(args.saved_verifier.parent / 'verify_official_kernel_tolerances.py', LOAD_HELPER_SHA),
        original_FA_tests=source_check(args.sources / 'test_flash_attn_v263.py', FA_SOURCE_SHA),
        row_wrapper=source_check(args.row_wrapper, ROW_WRAPPER_SHA),
        row_library=source_check(args.row_library, ROW_LIBRARY_SHA),
        scalar_wrapper=source_check(args.scalar_wrapper, SCALAR_WRAPPER_SHA),
        scalar_library=source_check(args.scalar_library, SCALAR_LIBRARY_SHA),
        environment=dict(path=str(args.environment_json), sha256=sha(args.environment_json)),
    )
    # No Torch, library or GPU import occurs merely by importing this helper.
    import torch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native = torch.load(args.native_operands, map_location='cpu', weights_only=True, mmap=True)
    saved = torch.load(args.finite_operands, map_location='cpu', weights_only=True, mmap=True)
    if saved['decoder_index'] != 3 or saved['invocation'] != 8:
        raise ValueError('Capture is not the actual eighth finite FA call at decoder3.')
    ops, outputs, packed = saved['operands'], saved['outputs'], native['tensors']
    if set(ops) != set(OPERANDS) or set(outputs) != set(OUTPUTS):
        raise ValueError('Actual saved finite operand/output schema changed.')
    lengths, starts = list(saved['lengths']), list(saved['query_starts'])
    coefficients = list(saved['coefficient_starts'])
    batch = len(lengths)
    if len(starts) != batch or len(coefficients) != batch or ops['q0'].shape[0] != batch:
        raise ValueError('Actual row layout does not identify every finite operand row.')
    cuq, cuk = [packed['packed_cu_seqlens_' + name] for name in ('q', 'k')]
    if cuq.shape != (2 * batch + 1,) or cuk.shape != (2 * batch + 1,):
        raise ValueError('Native capture must contain both actual endpoints for every finite row.')
    if ops['u'].dtype != torch.float32 or any(ops[name].dtype != torch.bfloat16
            for name in ('q0', 'q1', 'k0', 'k1', 'v0')):
        raise ValueError('Actual finite data is outside the original BF16/FP32 interface.')
    if any(ops[name].dtype != torch.float32 for name in ('lse0', 'lse1')):
        raise ValueError('Actual saved public LSE must remain FP32.')
    for row, (length, start, coefficient) in enumerate(zip(lengths, starts, coefficients)):
        count = length - start
        if not start <= coefficient < length:
            raise ValueError(f'Row {row} has no retained coefficient positions for the original seven-assertion verifier.')
        if not all(int(cuq[endpoint + 1] - cuq[endpoint]) == count and
                   int(cuk[endpoint + 1] - cuk[endpoint]) == length
                   for endpoint in (2 * row, 2 * row + 1)):
            raise ValueError('Actual native complete lengths disagree with the finite layout.')

    newer = module('_actual_row_finite_owner', args.row_wrapper)
    scalar = module('_original_actual_scalar_finite_owner', args.scalar_wrapper)
    original_layout, original_operation = newer.RightPaddedLengths, newer.VendorFAFiniteP1BF16D256

    class ActualRowLayout(original_layout):
        """Only transport the original verifier's scalar B1 cut into row ABI."""
        def __init__(self, valid_lengths, padded_length, device, **kwargs):
            start = kwargs.pop('query_start', 0)
            coefficients = kwargs.pop('coefficient_starts', None)
            super().__init__(valid_lengths, padded_length, device,
                coefficient_starts=coefficients,
                query_starts=[start] * len(valid_lengths),
                query_padded_length=padded_length - start, **kwargs)

    previous_module = sys.modules.get('vendor_fa_finite_bf16_d256')
    previous_argv, previous_path = sys.argv, list(sys.path)
    previous_environment = os.environ.get('DT_ENVIRONMENT_JSON')
    official_rows = []
    started = time.perf_counter()
    try:
        newer.RightPaddedLengths = ActualRowLayout
        newer.VendorFAFiniteP1BF16D256 = lambda _library, _sha: original_operation(args.row_library, ROW_LIBRARY_SHA)
        sys.modules['vendor_fa_finite_bf16_d256'] = newer
        sys.path.insert(0, str(args.saved_verifier.parent))
        os.environ['DT_ENVIRONMENT_JSON'] = str(args.environment_json)
        for row, (length, start, coefficient) in enumerate(zip(lengths, starts, coefficients)):
            count = length - start
            attention_values = {}
            for name in ('q', 'k', 'v'):
                value, cu = packed['packed_' + name], cuq if name == 'q' else cuk
                # Actual two endpoints, no reference token or tensor synthesized.
                attention_values['dense_' + name] = torch.stack([
                    value[int(cu[e]):int(cu[e + 1])] for e in (2 * row, 2 * row + 1)])
            payload = dict(attention_values=attention_values, fa=dict(
                operands=dict(u=ops['u'][row:row + 1, :, :count].contiguous()),
                lengths=[length], padded_length=length, query_start=start,
                coefficient_starts=[coefficient], scale=saved['scale']))
            row_path = args.output_dir / f'actual-complete-row{row}.pt'
            with row_path.open('xb') as stream:
                torch.save(payload, stream)
            output = args.output_dir / f'original-official-coincident-row{row}.json'
            sys.argv = [str(args.saved_verifier), '--operands', str(row_path),
                        '--sources', str(args.sources), '--output', str(output)]
            print(f'phase=original_unchanged_seven_assertions row={row} Q={count} K={length}', flush=True)
            runpy.run_path(str(args.saved_verifier), run_name='__main__')
            result = json.loads(output.read_bytes())
            if len(result['checks']) != 7:
                raise ValueError('Original seven-check verifier did not produce all seven checks.')
            official_rows.append(dict(row=row, complete_query_tokens=count, complete_key_tokens=length,
                factual_endpoint_index=2 * row + 1, original_query_start=start,
                original_coefficient_start=coefficient,
                representation=dict(path=str(row_path), sha256=sha(row_path)),
                original_result=dict(path=str(output), sha256=sha(output), status=result['status'])))
            del payload, attention_values
            gc.collect()
    finally:
        newer.RightPaddedLengths, newer.VendorFAFiniteP1BF16D256 = original_layout, original_operation
        sys.argv, sys.path = previous_argv, previous_path
        if previous_module is None:
            sys.modules.pop('vendor_fa_finite_bf16_d256', None)
        else:
            sys.modules['vendor_fa_finite_bf16_d256'] = previous_module
        if previous_environment is None:
            os.environ.pop('DT_ENVIRONMENT_JSON', None)
        else:
            os.environ['DT_ENVIRONMENT_JSON'] = previous_environment

    print('phase=actual_nonzero_scalar_vs_saved_row_outputs', flush=True)
    original_owner = scalar.VendorFAFiniteP1BF16D256(args.scalar_library, SCALAR_LIBRARY_SHA)
    nonzero_rows = []
    with torch.no_grad():
        for row, (length, start, coefficient) in enumerate(zip(lengths, starts, coefficients)):
            count = length - start
            row_ops = {name: value[row:row + 1, :, :length if name in ('k0', 'k1', 'v0') else count]
                       .contiguous().to('cuda') for name, value in ops.items()}
            layout = scalar.RightPaddedLengths([length], length, 'cuda',
                coefficient_starts=[coefficient], query_start=start)
            actual = original_owner(row_ops, saved['scale'], layout)
            checks = {}
            for name in OUTPUTS:
                expected = outputs[name][row:row + 1, :, :count]
                observed = actual[name].to('cpu')
                checks[name] = dict(equal=torch.equal(expected, observed),
                    max_abs=float((expected.float() - observed.float()).abs().max()),
                    shape=list(observed.shape), dtype=str(observed.dtype))
            nonzero_rows.append(dict(row=row, complete_query_tokens=count,
                complete_key_tokens=length, query_start=start, coefficient_start=coefficient,
                same_original_retained_range=True, checks=checks))
            del row_ops, actual, layout
    torch.cuda.synchronize()
    report = dict(scope=__doc__, source_checks=sources,
        native_operands=dict(path=str(args.native_operands), sha256=sha(args.native_operands)),
        finite_operands=dict(path=str(args.finite_operands), sha256=sha(args.finite_operands)),
        helper_source=dict(path=__file__, sha256=sha(__file__)),
        official_complete_B1_rows=official_rows,
        official_status='passed' if all(row['original_result']['status'] == 'passed' for row in official_rows) else 'failed',
        nonzero_scalar_vs_saved_B4_rows=nonzero_rows,
        nonzero_all_equal=all(item['equal'] for row in nonzero_rows for item in row['checks'].values()),
        elapsed_seconds=time.perf_counter() - started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        reference='Unmodified saved 7ff script, original dense FA assertions on each full logical B1 row',
        limits=['This does not replace the complete B8 native varlen forward assertion',
                'Official finite checks are coincident derivative-limit checks only',
                'Nonzero equality/max_abs is representation evidence, not a new finite tolerance'],
        model_loads=0, checkpoint_operations=0, random_operands_generated=False,
        official_source_or_assertions_changed=False, formal_changes=False)
    path = args.output_dir / 'actual-row-official-and-nonzero.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    if report['official_status'] != 'passed' or not report['nonzero_all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
