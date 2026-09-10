"""Gold-independent sentence ranking experiments; no attribution is changed."""
import re

import numpy as np


RULES = ('raw', 'mean', 'rms', 'max')


def sentence_pool_order(prompt_text, offsets, scores, keep, rule):
    if rule not in RULES:
        raise ValueError('unknown pooling rule')
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
    view = np.maximum(scores, 0)
    if rule == 'raw':
        return keep[np.lexsort((keep, -view[keep]))].tolist()
    boundaries = [m.end() for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+', prompt_text)]
    groups = np.searchsorted(boundaries, offsets[:, 0], side='right')
    count = np.bincount(groups[keep], minlength=len(boundaries) + 1)
    values = view[keep].astype(np.float64)
    if rule == 'max':
        pooled = np.zeros(len(boundaries) + 1, dtype=np.float64)
        np.maximum.at(pooled, groups[keep], values)
    else:
        mass = np.bincount(groups[keep], weights=values if rule == 'mean' else values ** 2,
                           minlength=len(boundaries) + 1)
        pooled = mass / np.maximum(count, 1)
        if rule == 'rms':
            pooled = np.sqrt(pooled)
    return keep[np.lexsort((keep, -view[keep], -pooled[groups[keep]]))].tolist()
