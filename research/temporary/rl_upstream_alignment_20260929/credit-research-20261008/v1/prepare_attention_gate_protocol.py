"""Bind a passive gate ledger to the measured final-attention discrepancy.

The existing matched mixer residual cannot distinguish sigmoid rounding,
product background mismatch and attention routing. This protocol adds those
readouts on the same frozen queries; it introduces no training candidate.
"""
import hashlib
import argparse
import json
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention-input', action='store_true')
    args = parser.parse_args()
    oF, oR, oD, sF, sD, dg, do, pF, pD, mc, mg, mp, cBF, cSmooth, dsSmooth = sp.symbols(
        'o_F o_R o_D s_F s_D delta_g delta_o p_F p_D m_content m_gate m_product c_BF c_smooth delta_s_smooth')
    ds = sF-sD
    terms = dict(
        coefficient_recompute=(mc-mp*sF)*do+(mg-mp*oR*cBF)*dg,
        product_background=mp*(oR-oD)*ds,
        sigmoid_rounded_secant=mp*oR*(cBF-cSmooth)*dg,
        sigmoid_background=mp*oR*(cSmooth*dg-dsSmooth),
        native_sigmoid_rounding=mp*oR*(dsSmooth-ds),
        native_product_rounding=mp*(oF*sF-oD*sD-(pF-pD)))
    gate = mc*do+mg*dg-mp*(pF-pD)
    residual = sp.expand((sum(terms.values())-gate).subs(do, oF-oD))
    assert residual == 0
    plan_path = HERE/'layer-collection-inputs.json'
    plan = json.loads(plan_path.read_bytes())
    paths = json.loads((HERE/'saved-propagation-order.json').read_bytes())
    cfg = json.loads((HERE/'gdn-owner-readonly.json').read_bytes())['model_config']['text_config']
    assert cfg['layer_types'][31] == 'full_attention'
    costs = {}
    sources = [plan_path, HERE/'saved-propagation-order.json', HERE/'gdn-owner-readonly.json',
        HERE/'passive_attention_gate.py', HERE/'passive_suboperations.py', HERE/'inspect_layer_collection.py',
        HERE/'saved-source-mask-audit/result.json', Path(__file__)]
    for task, spec in plan['tasks'].items():
        ranks = []
        for rank in (0,1):
            path = HERE/f'layer-suboperations-{task}-v2-observations/rank{rank}.json'
            sources.append(path)
            record = json.loads(path.read_bytes())
            assert record['phase'] == 'complete'
            ranks.append(record)
        width = max(e['selected_tokens'] for e in spec['entries'])
        bh = 4*width*cfg['hidden_size']
        costs[task] = dict(
            unchanged_frozen_queries=spec['unique_queries'], original_DT_calls_per_rank=[r['operations']['DT'] for r in ranks],
            original_native_calls_per_rank=[r['operations']['native_forward'] for r in ranks],
            original_measured_wall_seconds_per_rank=[r['elapsed_seconds'] for r in ranks],
            longest_actual_frozen_context=width,
            added_persistent_gate_host_bank_upper_bytes=48*bh,
            added_native_capture_host_upper_bytes=20*bh,
            projection_extra_device_tensor_upper_bytes=4*128*cfg['hidden_size']*14,
            projection_bound_scope='FP32 results/base+adapter sum and BF16 operand, excluding vendor GEMM workspace. '
                'Readout uses the original transpose in128-position slices; recomputation drift is measured.',
            maximum_projection_slices_per_DT=(width+127)//128,
            additional_model_calls_beyond_same_original_protocol=0,
            additional_DT_calls_beyond_same_original_protocol=0,
            added_scalar_work='Native sigmoid re-evaluation on actual operands and CPU FP64 ledger in128-position slices.',
            cost_scope='Conservative tensor-shape bounds and earlier measurements, not observed new peak or promised speed.')
        if args.attention_input:
            costs[task].update(additional_QKV_host_bank_upper_bytes=54*bh,
                additional_native_QKV_host_upper_bytes=48*bh,
                additional_device_model_or_kernel_calls_for_input_ledger=0,
                input_readout_scope='CPU copies of original endpoints and original dq/dk/dv; FP64 contractions in128-position slices. '
                    'Bounds assume compact KV width no greater than Q width, and include the original full-context cached K/V.')
    if args.attention_input:
        sources += [HERE/'attention-gate-textcraft/analysis.json', HERE/'attention-gate-textcraft/launch.json']
    result = dict(scope=__doc__, inputs=[ref(p) for p in sources],
        unchanged_queries_plan_sha256=ref(plan_path)['sha256'], selected_decoder_layers=[31],
        required_original_boundaries=[31,32], costs=costs,
        existing_evidence={task: v['robust_spurious_tail_summary']['matched'] for task,v in paths['tasks'].items()},
        arithmetic=dict(gate_error=str(gate), terms={k:str(v) for k,v in terms.items()},
            symbolic_residual=str(residual), smooth_sigmoid_secant_bound='.25 in real arithmetic; not a BF16 tolerance'),
        observations=[
            'Call each original boundary and return its exact result object; leave observer=None and consume_captures enabled.',
            'Reuse the original passive native FA capture and its selected Python events; do not replace model or FA functions.',
            'Preserve compiled-coefficient versus independent same-owner projection differences explicitly.',
            'Check recomputed native BF16 sigmoid/product against the captured actual product; report discrepancy without correction.',
            'Compare unchanged original DT source values against saved baseline and retain native factual drift.',
            'Separate frozen uniform/predicted-tail cohorts, prediction/native cells and exposure. Keep actual counts and conditional quantiles.',
            'The ledger measures terms applied to actual single-deletion changes. It is not a causal ablation or an instruction to subtract a term.',
            'Use TextCraft once first; judge observed phase cost/data before AppWorld. Do not rerun failed FA/GDN candidates.'
        ], numerical_rules_changed=False, method_changed=False, production_modified=False,
        official_tolerance_test=False, candidate=False, candidate_accepted=False,
        scope_limit='No kernel was changed. Algebraic closure and native product equality do not establish DT quality. '
            'Original author cumulative deletion/RISE/MAS remains the independent candidate quality check.',
        prepared_only=True, launched=False, wall_budget_per_worker_seconds=1800)
    if args.attention_input:
        n, g, qkv, content = sp.symbols('input_contraction gate_contraction QKV_contraction content_contraction')
        closure = sp.expand((n-g-qkv)+(qkv-content)-(n-g-content))
        assert closure == 0
        result.update(read_attention_input=True,
            core_decomposition=dict(input_projection_QKnorm_RoPE=str(n-g-qkv), finite_FA_core=str(qkv-content), symbolic_residual=str(closure)),
            reason='The previous passive run left core/input largest at nine paired-unchanged robust spurious positions across five states. '
                'Read actual original dq/dk/dv and endpoints to separate that measured remainder; no new numerical candidate.',
            prior_diagnostic_drift='Rank1 prior cross-process endpoint drift is preserved, unresolved and not erased. '
                'This ledger diagnoses same-call contractions; comparison with prior results must separate exact unchanged and changed points.')
    path = HERE/('attention-input-protocol.json' if args.attention_input else 'attention-gate-protocol.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=ref(path), symbolic_residual=str(residual), costs=costs)))


if __name__ == '__main__':
    main()
