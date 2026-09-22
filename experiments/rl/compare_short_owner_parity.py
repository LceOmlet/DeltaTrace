"""Summarize saved real-worker owner comparisons; no training implementation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value.detach().double().reshape(-1)]
    if isinstance(value, dict):
        return [t for key in sorted(value) for t in tensors(value[key])]
    if isinstance(value, list):
        return [t for item in value for t in tensors(item)]
    raise TypeError(type(value))


def difference(actual, reference):
    x, y = torch.cat(tensors(actual)), torch.cat(tensors(reference))
    delta = x-y
    return dict(elements=x.numel(), changed_elements=int((delta != 0).sum()),
                max_abs=float(delta.abs().max()),
                absolute_l2=float(delta.norm()),
                relative_l2=float(delta.norm()/y.norm().clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trim-shared-padding', action='store_true')
    parser.add_argument('--case-prefix', help='Receipt filename prefix when retaining before/after repair cases')
    parser.add_argument('--fp32-reference', type=Path,
                        help='Independent FP32 owner artifact; borrow the FA 2x error comparison for this local pipeline test')
    args = parser.parse_args()
    result = dict(scope='Same worker, weights, original optimizer and DT advantages; '
                        'installed actor versus pristine pinned VERL actor. '
                        'FA/math calibration uses only real action positions.',
                  owner_tolerance=('owner FA/math max-absolute and L2 error, no multiplier'
                                   if args.trim_shared_padding else dict(atol=0, rtol=0)), cases={})
    fp32_state = fp32_receipt = None
    if args.fp32_reference:
        assert args.trim_shared_padding
        fp32_state = torch.load(args.fp32_reference, map_location='cpu', weights_only=True)
        fp32_receipt = json.loads(args.fp32_reference.with_suffix('.json').read_text())
        assert fp32_receipt['status'] == 'artifacts_ready' and fp32_receipt['fp32_reference']
        result['owner_tolerance'] = 'max absolute error to independent FP32 <= 2x original BF16 math pipeline error to FP32'
        result['tolerance_source'] = 'https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/tests/test_flash_attn.py'
        result['tolerance_scope'] = ('Local whole-pipeline extension borrowing the FA error-comparison form. '
                                     'The denominator includes weight rounding, other layers and update accumulation. '
                                     'This is not an official FA test or a certification of PPO training reliability.')
        result['status_scope'] = 'Only the stated short-chain numerical error bound; not training reliability'
        result['fp32_reference'] = dict(receipt=fp32_receipt,
            artifacts_sha256=hashlib.sha256(args.fp32_reference.read_bytes()).hexdigest())
    for setting in ('reshard', 'noreshard'):
        prefix = args.case_prefix or ('short-parity-'+('trimmed-' if args.trim_shared_padding else 'paired-'))
        stem = args.receipts/(prefix+setting)
        receipt = json.loads(stem.with_suffix('.json').read_text())
        if args.trim_shared_padding:
            assert receipt['status'] == 'artifacts_ready' and receipt['trim_shared_padding'], receipt
        else:
            assert receipt['status'] == 'passed' and receipt['comparison_performed'], receipt
        state = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
        errors = {}
        for key, expected in state['paired_owner'].items():
            if not args.trim_shared_padding:
                torch.testing.assert_close(state[key], expected, atol=0, rtol=0)
            errors[key] = difference(state[key], expected)
        for step in range(2):
            key = 'after_'+str(step)
            actual = {k: v.double()-state['before'][k].double() for k, v in state[key].items()}
            expected = {k: v.double()-state['before'][k].double()
                        for k, v in state['paired_owner'][key].items()}
            errors['parameter_delta_'+str(step)] = difference(actual, expected)
        attention = state['attention_check']
        mask = state['response_mask'].bool()
        torch.testing.assert_close(attention['fa_before'][mask], attention['fa_after'][mask], atol=0, rtol=0)
        result['cases'][setting] = dict(
            status='passed', errors=errors, receipt=receipt,
            source=str(stem.with_suffix('.json')),
            receipt_sha256=hashlib.sha256(stem.with_suffix('.json').read_bytes()).hexdigest(),
            artifacts_sha256=hashlib.sha256(stem.with_suffix('.pt').read_bytes()).hexdigest(),
            fa_math_log_prob_error=difference(attention['fa_before'][mask], attention['math'][mask]),
            fa_repeat_log_prob_error=difference(attention['fa_before'][mask], attention['fa_after'][mask]),
            active_token_count=int(mask.sum()),
        )
        if args.trim_shared_padding:
            # Padding carries no policy loss. Compare the original active
            # columns and complete gradients/parameter increments separately.
            def policy_quantities(values):
                outputs = dict(old_log_probs=values['old_log_probs'][mask],
                               policy_forward_log_probs=torch.cat(values['policy_forward_log_probs'])[mask.repeat(2, 1)],
                               raw_gradients=values['raw_gradients'])
                for step in range(2):
                    outputs['parameter_delta_'+str(step)] = {
                        k: v.double()-state['before'][k].double()
                        for k, v in values['after_'+str(step)].items()}
                return outputs
            actual = policy_quantities(state)
            reference = policy_quantities(state['paired_owner'])
            math_reference = policy_quantities(state['paired_owner_math'])
            bounded = {}
            for key in actual:
                error = difference(actual[key], reference[key])
                budget = difference(reference[key], math_reference[key])
                bounded[key] = dict(error=error, owner_fa_math_error=budget,
                                    passed=error['max_abs'] <= budget['max_abs'] and
                                           error['absolute_l2'] <= budget['absolute_l2'])
            result['cases'][setting]['fa_bounded_comparison'] = bounded
            result['cases'][setting]['status'] = ('passed' if all(v['passed'] for v in bounded.values()) else 'failed')
            if fp32_state is not None:
                for key in ('input_ids', 'attention_mask', 'responses', 'response_mask',
                            'dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates'):
                    torch.testing.assert_close(state[key], fp32_state[key], atol=0, rtol=0)
                torch.testing.assert_close({k:v.float() for k,v in state['before'].items()},
                                           fp32_state['before'], atol=0, rtol=0)
                fp32 = policy_quantities(fp32_state)
                fa_standard = {}
                for key in actual:
                    error = difference(actual[key], fp32[key])
                    baseline = difference(math_reference[key], fp32[key])
                    fa_standard[key] = dict(error_to_fp32=error, baseline_error_to_fp32=baseline,
                                            passed=error['max_abs'] <= 2*baseline['max_abs'])
                for key in ('actor/pg_loss', 'actor/pg_clipfrac', 'actor/pg_clipfrac_lower'):
                    candidate = torch.tensor([step[key] for step in receipt['updates']], dtype=torch.float64)
                    baseline_values = torch.tensor([step[key] for step in receipt['paired_math_updates']], dtype=torch.float64)
                    exact = torch.tensor([step[key] for step in fp32_receipt['updates']], dtype=torch.float64)
                    error, baseline = difference(candidate, exact), difference(baseline_values, exact)
                    fa_standard[key] = dict(error_to_fp32=error, baseline_error_to_fp32=baseline,
                                            passed=error['max_abs'] <= 2*baseline['max_abs'])
                result['cases'][setting]['fa_standard_comparison'] = fa_standard
                result['cases'][setting]['status'] = ('passed' if all(v['passed'] for v in fa_standard.values()) else 'failed')
    result['status'] = 'passed' if all(v['status'] == 'passed' for v in result['cases'].values()) else 'failed'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: dict(status=v['status'], max_owner_error=max(e['max_abs'] for e in v['errors'].values()),
                             fa_math=v['fa_math_log_prob_error']) for k, v in result['cases'].items()}))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
