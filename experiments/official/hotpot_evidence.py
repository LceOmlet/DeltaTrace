"""Native HotpotQA evidence units and explicitly budgeted supporting-fact scores.

Candidate construction and ranking have no gold, answer, or target argument.
The frozen retrieval_views module remains available for historical replay.
"""
import bisect
import math
import re

import numpy as np


def native_units(prompt, contexts, source_start, source_end):
    """Map complete source documents, then their native sentence-array offsets."""
    bodies = []
    seen = set()
    for title, sentences in contexts:
        if title in seen:
            raise ValueError('Duplicate context title')
        seen.add(title)
        document = title + '\n' + ''.join(sentences)
        matches = list(re.finditer(re.escape(document), prompt))
        if len(matches) != 1:
            raise ValueError(f'Complete context document must occur exactly once: {title}')
        start = matches[0].start() + len(title) + 1
        for index, sentence in enumerate(sentences):
            end = start + len(sentence)
            if end > start:
                if not source_start <= start < end <= source_end:
                    raise ValueError('Context sentence outside source')
                assert prompt[start:end] == sentence
                bodies.append(dict(kind='sentence', title=title, sentence_index=index,
                                   start=start, end=end))
            start = end
    bodies.sort(key=lambda x: x['start'])
    units, cursor = [], source_start
    for body in bodies:
        if body['start'] < cursor:
            raise ValueError('Overlapping native sentences')
        if body['start'] > cursor:
            units.append(dict(kind='metadata', title=None, sentence_index=None,
                              start=cursor, end=body['start']))
        units.append(body)
        cursor = body['end']
    if cursor < source_end:
        units.append(dict(kind='metadata', title=None, sentence_index=None,
                          start=cursor, end=source_end))
    if not bodies:
        raise ValueError('No native sentences')
    return units


def token_groups(text, offsets, keep, units, *, coordinate_shift=1):
    """Assign each eligible token exactly once using its first content character.

Native unit coordinates refer to the original prompt; offsets commonly refer
to ' '+prompt. Pure whitespace tokens use their start and remain charged.
Reject tokens that cross a unit boundary with non-whitespace on both sides.
"""
    starts = [u['start'] + coordinate_shift for u in units]
    ends = [u['end'] + coordinate_shift for u in units]
    groups = [[] for _ in units]
    for token in sorted(set(map(int, keep))):
        start, end = map(int, offsets[token])
        if not 0 <= start < end <= len(text):
            raise ValueError('Invalid token offset')
        anchor = start
        while anchor < end and text[anchor].isspace():
            anchor += 1
        if anchor == end:
            anchor = start
        group = bisect.bisect_right(starts, anchor) - 1
        if group < 0 or anchor >= ends[group]:
            raise ValueError('Eligible token outside unit coverage')
        if end > ends[group] and text[ends[group]:end].strip():
            raise ValueError('One token has content in multiple native units')
        groups[group].append(token)
    assert sorted(t for g in groups for t in g) == sorted(set(map(int, keep)))
    return groups


def regex_order(text, offsets, scores, keep, *, content_anchor):
    """Ablation of the old regex scorer with only the whitespace anchor fixed."""
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    eligible = np.asarray(sorted(set(map(int, keep))), dtype=np.int64)
    boundaries = [m.end() for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+', text)]
    anchors = []
    for start, end in offsets:
        a = start
        if content_anchor:
            while a < end and text[a].isspace():
                a += 1
            if a == end:
                a = start
        anchors.append(a)
    groups = np.searchsorted(boundaries, anchors, side='right')
    count = np.bincount(groups[eligible], minlength=len(boundaries) + 1)
    mass = np.bincount(groups[eligible], weights=values[eligible], minlength=len(boundaries) + 1)
    means = mass / np.maximum(count, 1)
    return eligible[np.lexsort((eligible, -values[eligible], -means[groups[eligible]]))].tolist()


def native_token_order(scores, groups):
    """Native-unit mean ranking, retaining the original exact token budget."""
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite attribution')
    token_mean = {}
    for group in groups:
        if group:
            mean = float(np.sum(values[group], dtype=np.float64) / len(group))
            token_mean.update((token, mean) for token in group)
    return sorted(token_mean, key=lambda t: (-token_mean[t], -float(values[t]), t))


def sentence_order(scores, units, groups):
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite attribution')
    candidates = [i for i,u in enumerate(units) if u['kind'] == 'sentence' and groups[i]]
    means = {i: float(np.sum(values[groups[i]], dtype=np.float64) / len(groups[i])) for i in candidates}
    return sorted(candidates, key=lambda i: (-means[i], units[i]['start']))


def select_sentences(order, groups, *, token_budget=None, sentence_budget=None):
    if (token_budget is None) == (sentence_budget is None):
        raise ValueError('Choose exactly one budget unit')
    if len(set(order)) != len(order) or any(not groups[i] for i in order):
        raise ValueError('Invalid sentence ranking')
    if sentence_budget is not None:
        if sentence_budget < 0:
            raise ValueError('Negative budget')
        selected = order[:sentence_budget]
    else:
        if token_budget < 0:
            raise ValueError('Negative budget')
        selected, remaining = [], token_budget
        for i in order:
            cost = len(groups[i])
            if cost <= remaining:
                selected.append(i)
                remaining -= cost
    tokens = [t for i in selected for t in groups[i]]
    assert len(tokens) == len(set(tokens))
    if token_budget is not None:
        assert len(tokens) <= token_budget
    return selected, tokens


def supporting_fact_metrics(selected, units, gold_keys):
    """Official set precision/Recall/F1/EM; full-support hit allows extra facts."""
    predicted = {(units[i]['title'], units[i]['sentence_index']) for i in selected}
    gold = set(map(tuple, gold_keys))
    if not gold:
        raise ValueError('Empty supporting-fact gold')
    all_keys = {(u['title'],u['sentence_index']) for u in units if u['kind']=='sentence'}
    if not gold <= all_keys:
        raise ValueError('Supporting-fact key has no source unit')
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.
    recall = true_positive / len(gold)
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                exact_match=float(predicted == gold), complete_support=float(gold <= predicted),
                true_positive=true_positive, predicted=len(predicted), gold=len(gold))


def fact_recall_ceiling(units, groups, gold_keys, *, token_budget=None, sentence_budget=None):
    gold = set(map(tuple, gold_keys))
    costs = sorted(len(groups[i]) for i,u in enumerate(units)
                   if u['kind']=='sentence' and (u['title'],u['sentence_index']) in gold)
    if len(costs) != len(gold) or not all(costs):
        raise ValueError('Unretrievable gold fact')
    if sentence_budget is not None:
        return min(len(gold), sentence_budget)/len(gold)
    count = 0
    for cost in costs:
        if cost > token_budget:
            break
        count += 1
        token_budget -= cost
    return count/len(gold)
