"""CPU diagnostics for fixed-budget token recovery; no changes to official scores.

Recall@f has ceiling min(1, ceil(f * eligible_count) / eligible_gold_count).
Report both the released recall and the ceiling-adjusted hit rate. The latter
is descriptive and must not replace the released benchmark column.
"""
import math

import numpy as np


def recovery_diagnostics(scores, keep, gold, fraction=0.1):
    """Return deterministic recall plus tie bounds and budget diagnostics.

    Inputs must already be the intended score view (e.g. positive or absolute).
    Equal scores are ordered by prompt position; bounds expose when an author's
    torch.topk selection can legitimately differ from this CPU tie order.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite one-dimensional vector')
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be finite and between zero and one')
    keep = np.asarray(sorted({int(i) for i in keep if 0 <= int(i) < len(scores)}), dtype=int)
    if not len(keep):
        raise ValueError('no eligible tokens')
    eligible_gold = set(map(int, gold)).intersection(keep.tolist())
    if not eligible_gold:
        raise ValueError('no eligible gold tokens')
    n, g = len(keep), len(eligible_gold)
    k = min(n, max(1, math.ceil(n * fraction)))
    order = keep[np.argsort(-scores[keep], kind='stable')]
    selected = order[:k]
    hit = len(set(selected.tolist()).intersection(eligible_gold))
    cutoff = float(scores[selected[-1]])
    above = keep[scores[keep] > cutoff]
    equal = keep[scores[keep] == cutoff]
    above_gold = len(set(above.tolist()).intersection(eligible_gold))
    equal_gold = len(set(equal.tolist()).intersection(eligible_gold))
    slots = k - len(above)
    low = above_gold + max(0, slots - (len(equal) - equal_gold))
    high = above_gold + min(slots, equal_gold)
    ceiling = min(1.0, k / g)
    return {
        'eligible': n, 'gold': g, 'budget': k,
        'gold_density': g / n, 'ceiling': ceiling,
        'random_expected_recall': k / n,
        'recall': hit / g, 'precision': hit / k,
        'ceiling_adjusted_recall': hit / min(k, g),
        'cutoff': cutoff, 'cutoff_tie_size': len(equal),
        'recall_tie_low': low / g, 'recall_tie_high': high / g,
        'selected': selected.tolist(),
    }


def reported_recovery_diagnostics(scores, keep, gold, reported_recall, fraction=0.1):
    """Attach budget context to an author's actual top-k result, preserving it."""
    d = recovery_diagnostics(np.maximum(np.asarray(scores, dtype=np.float32), 0), keep, gold, fraction)
    if not d['recall_tie_low'] - 1e-12 <= reported_recall <= d['recall_tie_high'] + 1e-12:
        raise ValueError('reported recall does not match the supplied scores/maps')
    d.pop('selected')
    d['deterministic_cpu_recall'] = d.pop('recall')
    d['reported_recall'] = float(reported_recall)
    d['ceiling_adjusted_recall'] = reported_recall / d['ceiling']
    d['precision'] = reported_recall * d['gold'] / d['budget']
    d['scope'] = 'diagnostic only; original Recall@10% is unchanged'
    return d
