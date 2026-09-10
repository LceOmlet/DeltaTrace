"""Optional structure-aware retrieval ranks, separate from signed attribution.

The selector sees only prompt offsets, eligible positions and token scores.
Gold labels and target text are intentionally absent from this API.
"""
import re

import numpy as np


def restrict_order_to_span(order, offsets, start, end):
    """Restrict a ranking to an explicit source span, preserving its order.

    The caller owns the task/source boundary and the token budget. This never
    infers a boundary from answer labels or attribution values. Token overlap
    follows the same half-open character-span convention as the benchmark.
    """
    if start < 0 or end <= start:
        raise ValueError('expected a nonempty half-open character span')
    return [i for i in order if offsets[i][0] < end and offsets[i][1] > start]


def sentence_density_order(prompt_text, offsets, scores, keep, *, absolute=False):
    """Rank eligible tokens by mean segment score, token score, then position.

    Segments end at newlines or sentence-ending punctuation plus whitespace.
    Offsets must refer to prompt_text, including any tokenizer leading space.
    This returns indices, not a conserved attribution vector.
    """
    scores = np.asarray(scores, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    if scores.ndim != 1 or offsets.shape != (len(scores), 2):
        raise ValueError('one offset pair is required per score')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    if (offsets < 0).any() or (offsets[:, 1] < offsets[:, 0]).any() or (offsets > len(prompt_text)).any():
        raise ValueError('invalid prompt offsets')
    keep = np.asarray(sorted({int(i) for i in keep}), dtype=np.int64)
    if len(keep) and (keep.min() < 0 or keep.max() >= len(scores)):
        raise ValueError('eligible index outside score vector')
    boundaries = [m.end() for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+', prompt_text)]
    groups = np.searchsorted(boundaries, offsets[:, 0], side='right')
    view = abs(scores) if absolute else np.maximum(scores, 0)
    count = np.bincount(groups[keep], minlength=len(boundaries) + 1)
    mass = np.bincount(groups[keep], weights=view[keep], minlength=len(boundaries) + 1)
    density = mass / np.maximum(count, 1)
    order = np.lexsort((keep, -view[keep], -density[groups[keep]]))
    return keep[order].tolist()
