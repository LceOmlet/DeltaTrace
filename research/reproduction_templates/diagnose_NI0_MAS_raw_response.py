"""Offline diagnosis of saved original MAS curves; never replaces the metric.

Run from any directory. Reads the published original curve evidence and writes
only numerical diagnostics. No model calls, new scores, or metric modifications.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'evidence/dt_mlp_original_metrics_NI0_20260908_v1/results.json'
DEST = ROOT / 'evidence/NI0_MAS_raw_response_diagnosis_20260908.json'


def auc(values):
    x = np.asarray(values, dtype=np.float64)
    return float((x.sum() - x[0] / 2 - x[-1] / 2) / (len(x) - 1))


def diagnose(row):
    score = np.asarray(row['scores'], dtype=np.float64)
    density = np.asarray(row['density'], dtype=np.float64)
    original = np.asarray(row['normalized_model_response'], dtype=np.float64)
    raw = (score - score[-1]) / abs(score[0] - score[-1])
    assert np.array_equal(original, np.minimum.accumulate(np.clip(raw, 0, 1)))
    assert np.allclose(np.abs(original - density), row['alignment_penalty'], rtol=0, atol=1e-15)
    groups = []
    for k in range(1, len(score)):
        old = set(row['input_receipts'][k - 1]['deleted_positions'])
        new = set(row['input_receipts'][k]['deleted_positions'])
        assert old < new
        predicted = float(density[k - 1] - density[k])
        actual = float(score[k - 1] - score[k])
        groups.append({
            'step': k, 'positions': sorted(new - old),
            'cumulative_deleted_count': len(new),
            'predicted_fraction_of_net_effect': predicted,
            'observed_conditional_logprob_effect': actual,
            'observed_fraction_of_net_effect': float(raw[k - 1] - raw[k]),
            'predicted_negative': predicted < -1e-8,
            'observed_sign': 'negative' if actual < 0 else ('positive' if actual > 0 else 'tie_at_native_precision'),
        })
    negative = [g for g in groups if g['predicted_negative']]
    original_ap = auc(np.abs(original - density))
    raw_ap = auc(np.abs(raw - density))
    return {
        'original_RISE': row['return_metrics'][0], 'original_MAS': row['return_metrics'][1],
        'original_alignment_AUC': original_ap, 'unclipped_response_alignment_AUC_diagnostic_only': raw_ap,
        'clipping_and_prefix_min_change_to_alignment_AUC': original_ap - raw_ap,
        'density_outside_unit_interval_AUC': auc(np.maximum(-density, 0) + np.maximum(density - 1, 0)),
        'raw_response': raw.tolist(), 'density': density.tolist(),
        'raw_response_below_all_EOS_steps': np.flatnonzero(raw < 0).tolist(),
        'raw_response_below_all_EOS_count': int(np.sum(raw < 0)),
        'negative_predicted_group_count': len(negative),
        'negative_predicted_group_observations': {s: sum(g['observed_sign'] == s for g in negative)
            for s in ['negative', 'positive', 'tie_at_native_precision']},
        'observed_negative_group_steps': [g['step'] for g in groups if g['observed_sign'] == 'negative'],
        'group_conditional_effects': groups,
    }


def main():
    raw = SOURCE.read_bytes()
    result = json.loads(raw)
    assert result['status'] == 'original_RISE_MAS_on_fixed_NI0_input_completed'
    rows = {name: diagnose(row) for name, row in result['curves'].items()}
    dt = rows['DT_frozen']
    # Decision-relevant observations, checked against recorded original curves.
    assert dt['raw_response_below_all_EOS_count'] == 0
    assert dt['clipping_and_prefix_min_change_to_alignment_AUC'] < 0
    assert dt['negative_predicted_group_observations'] == {'negative': 1, 'positive': 1, 'tie_at_native_precision': 6}
    report = {
        'status': 'offline_original_curve_diagnosis_complete',
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'sample_count': 1, 'dataset': 'author_processed_niah_mq_q2', 'index': 0,
        'model': 'Qwen3.5-9B', 'total_input_tokens': 588,
        'extra_model_calls': 0, 'FT_modified': False, 'DT_modified': False,
        'original_RISE_MAS_unchanged': True, 'methods': rows,
        'conclusions': [
            'The 45.2% outside-density portion is a geometric decomposition, not evidence that response clipping caused the MAS deficit.',
            'On this sample the actual unclipped DT deletion response never falls below the all-EOS endpoint. Removing original clipping/prefix-min slightly increases DT alignment error.',
            'DT static allocation magnitudes do not track this cumulative deletion path: the first group predicts 43.77% of net effect, versus 18.92% observed.',
            'One sample cannot identify population mean MAS, variance, prevalence of this mismatch, or a sample size needed to establish an advantage.',
        ],
        'limits': [
            'Raw-response alignment is a diagnostic only, not a replacement MAS or reported new benchmark.',
            'Native default scoring precision retained. Equal logprob values leave small effects unresolved; ties are not proof of zero effect.',
            'Group effects are conditional on the preceding cumulative deletions, not individual token sign ground truth or the original two-endpoint allocation.',
            'Twenty correlated deletion steps are not twenty independent samples. This historically used example is not a held-out confirmation set.',
            'An explicit external input formatter preserves the same raw input for FT/DT and original scoring; this is not stock full-runner or paper-table reproduction.',
        ],
    }
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'sample_count': 1, 'extra_model_calls': 0,
        'methods': {k: {x: v[x] for x in ['original_RISE', 'original_MAS', 'original_alignment_AUC',
          'unclipped_response_alignment_AUC_diagnostic_only', 'negative_predicted_group_observations']} for k, v in rows.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
