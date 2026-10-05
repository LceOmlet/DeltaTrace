"""CPU-only diagnosis of actual layer3 norm/storage operands, with native owners.

No model checkpoint, optimizer, finite replacement, correction, or acceptance
threshold is used. The actual installed HF RMSNorm and pinned c9 finite rules
are invoked on the captured tensors. CPU owner recomputation is identified
separately from the original GPU compiled return and native BF16 outputs.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import torch


def source_of(function):
    function = inspect.unwrap(function)
    path = Path(inspect.getsourcefile(function)).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        function=function.__name__, first_line=function.__code__.co_firstlineno)


def tensor_metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
        stride=list(value.stride()))


def raw_difference(actual, reference):
    # Raw diagnostics only; no allclose threshold or pass/fail decision.
    delta = actual.double()-reference.double()
    return dict(actual=tensor_metadata(actual), reference=tensor_metadata(reference),
        exactly_equal=torch.equal(actual, reference),
        maximum_absolute_difference=float(delta.abs().max()),
        rms_difference=float(delta.square().mean().sqrt()),
        mean_difference=float(delta.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--owner-root', type=Path,
        help='Pinned c9 clean/qwen35 directory; defaults to the captured decoder source directory.')
    args = parser.parse_args()
    payload = torch.load(args.input, map_location='cpu', weights_only=False)
    captured_receipt = payload['receipt']
    owner_root = (args.owner_root or Path(captured_receipt['owner_decoder']['path']).parent).resolve()
    sys.path.insert(0, str(owner_root))
    decoder = importlib.import_module('qwen35_decoder_finite')
    signed = importlib.import_module('signed_secant_rules')
    runner = importlib.import_module('qwen35_dense_finite_runner')
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    sources = dict(hf_forward=source_of(Qwen3_5RMSNorm.forward),
        hf_norm_primitive=source_of(Qwen3_5RMSNorm._norm),
        finite_norm_primitive=source_of(signed.rmsnorm_secant_pullback),
        finite_norm_residual=source_of(decoder._norm_residual_rule),
        token_effect=source_of(runner._token_effect))
    # Identity is a provenance contract, not a numerical acceptance gate.
    if sources['finite_norm_residual']['sha256'] != captured_receipt['owner_decoder']['sha256']:
        raise RuntimeError('Imported decoder source differs from the actual captured owner.')
    if Path(sources['finite_norm_residual']['path']).parent != owner_root:
        raise RuntimeError('The decoder was not imported from the requested owner directory.')

    def effect(coefficients, endpoints):
        return float(runner._token_effect(coefficients, endpoints).sum())

    native = payload['native']
    calls = {row['name']: row for row in payload['finite_norm_calls']}
    result = dict(scope=__doc__, backend='CPU on captured operands; original saved return was compiled on GPU',
        input_path=str(args.input.resolve()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        analyzer_path=str(Path(__file__).resolve()),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__, owner_root=str(owner_root), sources=sources,
        captured_receipt=captured_receipt, norms=[], residual_additions=[],
        interpretation=(
            'Each effect is the original _token_effect contraction. The captured GPU BF16 '
            'output is compared with CPU native forward and CPU continuous weighted _norm. '
            'Their difference is reported separately, so GPU/backend variation is not '
            'silently attributed entirely to BF16 storage. No number is an official '
            'FA/FLA/PPO accuracy assertion.'))

    with torch.no_grad():
        for name, input_field, output_field in (
                ('post_attention_norm', 'post_norm_input', 'post_norm_output'),
                ('input_norm', 'input_norm_input', 'input_norm_output')):
            call = calls[name]
            x0, x1 = call['x0'], call['x1']
            x = torch.cat((x0, x1), dim=0)
            weight = call['raw_weight']
            upstream, residual = call['upstream'], call['residual']
            saved = call['result']
            actual_input, actual_output = native[input_field], native[output_field]
            norm = Qwen3_5RMSNorm(x.shape[-1], eps=call['eps']).to(device='cpu', dtype=weight.dtype)
            norm.weight.copy_(weight)
            cpu_native_output = norm(x)
            # Reuse HF's existing normalization primitive. The same raw-weight
            # factor is applied in the requested FP32/FP64 diagnostic dtype;
            # HF.forward itself always normalizes in FP32 before type_as(x).
            continuous32 = norm._norm(x.float())*(1.0+norm.weight.float())
            continuous64 = norm._norm(x.double())*(1.0+norm.weight.double())
            eager = decoder._norm_residual_rule(x0, x1, weight, upstream, residual, call['eps'])
            primitive32 = signed.rmsnorm_secant_pullback(x0.float(), x1.float(),
                1.0+weight.float(), upstream.float(), call['eps'])
            primitive64 = signed.rmsnorm_secant_pullback(x0.double(), x1.double(),
                1.0+weight.double(), upstream.double(), call['eps'])
            effects = dict(
                captured_compiled_input_total=effect(saved, x),
                captured_residual=effect(residual, x),
                captured_native_BF16_output=effect(upstream, actual_output),
                same_owner_CPU_eager_input_total=effect(eager, x),
                finite_primitive_FP32_input=effect(primitive32, x),
                finite_primitive_FP64_input=effect(primitive64, x.double()),
                CPU_native_BF16_output=effect(upstream, cpu_native_output),
                CPU_continuous_FP32_output=effect(upstream, continuous32),
                CPU_continuous_FP64_output=effect(upstream.double(), continuous64))
            decomposition = dict(
                captured_compiled_gap=(effects['captured_compiled_input_total']-
                    effects['captured_residual']-effects['captured_native_BF16_output']),
                same_owner_CPU_eager_gap=(effects['same_owner_CPU_eager_input_total']-
                    effects['captured_residual']-effects['captured_native_BF16_output']),
                saved_compiled_minus_CPU_eager_total=(effects['captured_compiled_input_total']-
                    effects['same_owner_CPU_eager_input_total']),
                continuous_FP32_finite_gap=(effects['finite_primitive_FP32_input']-
                    effects['CPU_continuous_FP32_output']),
                continuous_FP64_finite_gap=(effects['finite_primitive_FP64_input']-
                    effects['CPU_continuous_FP64_output']),
                CPU_eager_residual_addition_gap=(effects['same_owner_CPU_eager_input_total']-
                    effects['captured_residual']-effects['finite_primitive_FP32_input']),
                continuous_FP32_minus_FP64_output=(effects['CPU_continuous_FP32_output']-
                    effects['CPU_continuous_FP64_output']),
                CPU_BF16_storage_minus_continuous_FP32_output=(effects['CPU_native_BF16_output']-
                    effects['CPU_continuous_FP32_output']),
                captured_GPU_BF16_minus_CPU_BF16_output=(effects['captured_native_BF16_output']-
                    effects['CPU_native_BF16_output']),
                captured_GPU_BF16_minus_CPU_continuous_FP32_output=(effects['captured_native_BF16_output']-
                    effects['CPU_continuous_FP32_output']))
            result['norms'].append(dict(name=name, eps=call['eps'],
                raw_tensors={key: tensor_metadata(call[key])
                    for key in ('x0', 'x1', 'raw_weight', 'upstream', 'residual', 'result')},
                native_captured_input_vs_finite_input=raw_difference(actual_input, x),
                captured_native_output_vs_same_HF_CPU_forward=raw_difference(actual_output, cpu_native_output),
                saved_compiled_vs_same_owner_CPU_eager=raw_difference(saved, eager),
                eager_minus_residual_vs_finite_primitive_FP32=raw_difference(eager-residual, primitive32),
                effects=effects, decomposition=decomposition))

        for name, norm_name in (
                ('attention_residual_add', 'input_norm'),
                ('mlp_residual_add', 'post_attention_norm')):
            addition = payload['residual_additions'][name]
            coefficients = calls[norm_name]['residual']
            left, right, output = addition['left'], addition['right'], addition['output']
            left_effect, right_effect, output_effect = (
                effect(coefficients, value) for value in (left, right, output))
            cpu_BF16_sum = left+right
            cpu_FP32_sum = left.float()+right.float()
            cpu_BF16_effect, cpu_FP32_effect = (
                effect(coefficients, value) for value in (cpu_BF16_sum, cpu_FP32_sum))
            result['residual_additions'].append(dict(name=name,
                coefficient_from=name+' uses '+norm_name+'.residual',
                coefficient=tensor_metadata(coefficients),
                tensors={key: tensor_metadata(addition[key]) for key in ('left', 'right', 'output')},
                original_fields=addition['original_fields'],
                actual_output_vs_CPU_same_dtype_addition=raw_difference(output, cpu_BF16_sum),
                effects=dict(actual_output=output_effect, left_branch=left_effect, right_branch=right_effect,
                    CPU_BF16_sum=cpu_BF16_effect, CPU_FP32_sum=cpu_FP32_effect),
                decomposition=dict(actual_output_minus_branch_sum=output_effect-left_effect-right_effect,
                    CPU_BF16_storage_minus_FP32_sum=cpu_BF16_effect-cpu_FP32_effect,
                    actual_output_minus_CPU_BF16_sum=output_effect-cpu_BF16_effect,
                    CPU_FP32_sum_minus_branch_sum=cpu_FP32_effect-left_effect-right_effect)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        norm_count=len(result['norms']), addition_count=len(result['residual_additions']),
        backend=result['backend'])))


if __name__ == '__main__':
    main()
