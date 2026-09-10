"""Versioned evidence-source selection and recovery evaluation.

Source parsing sees only the task name and prompt. Gold is used afterwards to
check coverage and score predictions. No boundary is fitted to labels/scores.
"""
import hashlib
import re

import numpy as np

from recovery_diagnostics import recovery_diagnostics

SOURCE_PROTOCOL = 'source-v2'
LEGACY_PROTOCOL = 'released-v1'
SUPPORTED_TASKS = (
    'niah_mq_q2', 'niah_mq_q4', 'niah_mq_q8', 'niah_mv_v2', 'niah_mv_v4', 'niah_mv_v8',
    'vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long',
)
NIAH_INSTRUCTION = 'Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.'
NIAH_QUERY = '\nWhat are all the special magic numbers for '
VT_INSTRUCTION = 'Memorize and track the chain(s) of variable assignment hidden in the following text.'
HP_INSTRUCTION = 'Answer the question based on the given documents. Only give me the answer and do not output any other words.'
HP_DOCUMENTS = 'The following are given documents.'


def configure_run(args, release):
    """Resolve defaults and reject incomparable protocols before loading a model."""
    source = args.evaluation_protocol == SOURCE_PROTOCOL
    if args.evaluation_protocol not in (SOURCE_PROTOCOL, LEGACY_PROTOCOL):
        raise ValueError('Unknown evaluation protocol')
    if source and args.selection == 'development16':
        raise ValueError('development16 is the historical NI/MH fixture; use released-v1, or source-v2 smoke/paper.')
    if args.sentence_recovery and not source:
        raise ValueError('Sentence recovery is a separate source-v2 evaluation; released-v1 stays unchanged.')
    if args.datasets is None:
        args.datasets = list(SUPPORTED_TASKS) if source else (
            list(release['tasks']) if args.selection == 'paper' else ['niah_mq_q2', 'morehopqa'])
    if not args.datasets or len(set(args.datasets)) != len(args.datasets) or any(t not in release['tasks'] for t in args.datasets):
        raise ValueError('Datasets must be unique tasks from the released table.')
    if source and any(t not in SUPPORTED_TASKS for t in args.datasets):
        raise ValueError('source-v2 supports the 11 evidence-recovery tasks; other tasks require released-v1.')
    if not source and args.selection != 'paper' and not set(args.datasets) <= {'niah_mq_q2', 'morehopqa'}:
        raise ValueError('Legacy development/smoke only select the frozen NI and MH tasks.')
    if args.ft is None:
        args.ft = 'published' if not source and args.family == 'qwen3' and args.selection == 'paper' else 'live'
    if args.ft == 'published' and (source or args.family != 'qwen3' or args.selection != 'paper'):
        raise ValueError('Published FT means require released-v1, Qwen3, and a complete paper task; source-v2 requires live FT.')


def source_span(dataset, prompt):
    """Parse the fixed released scaffold; refuse unknown or malformed templates."""
    if dataset not in SUPPORTED_TASKS:
        raise ValueError(f'No source-v2 parser for {dataset}')
    if dataset.startswith('niah_'):
        if not prompt.startswith(NIAH_INSTRUCTION + '\n') or prompt.count(NIAH_QUERY) != 1:
            raise ValueError('Unrecognized NIAH instruction/query scaffold')
        start, end = len(NIAH_INSTRUCTION) + 1, prompt.index(NIAH_QUERY)
        kind = 'niah_body'
    elif dataset.startswith('vt_'):
        headers = list(re.finditer(r'(?m)^' + re.escape(VT_INSTRUCTION) + r'$', prompt))
        if len(headers) != 2 or headers[0].start() != 0:
            raise ValueError('Expected exactly one solved VT demonstration and one current problem')
        start = headers[1].end()
        query = list(re.finditer(r'(?m)^Question: ', prompt[start:]))
        if len(query) != 1:
            raise ValueError('Unrecognized VT current-question boundary')
        end = start + query[0].start()
        kind = 'vt_current_body'
    else:
        headers = list(re.finditer(r'(?m)^' + re.escape(HP_INSTRUCTION) + r'$', prompt))
        docs = list(re.finditer(r'(?m)^' + re.escape(HP_DOCUMENTS) + r'$', prompt))
        if len(headers) != 2 or headers[0].start() != 0 or len(docs) != 1:
            raise ValueError('Unrecognized HotpotQA document scaffold')
        start, end = docs[0].end(), headers[1].start()
        if not re.match(r'\s*Document 1:', prompt[start:end]) or not re.match(r'\s*Question: ', prompt[headers[1].end():]):
            raise ValueError('Unrecognized HotpotQA document/question boundary')
        kind = 'hotpot_documents'
    # Keep scaffold-delimited whitespace: a leading-space BPE token can contain
    # the first evidence word. Stripping characters would incorrectly lose it.
    if not 0 <= start < end <= len(prompt) or not prompt[start:end].strip():
        raise ValueError('Empty or invalid evidence body')
    return {'kind': kind, 'start': start, 'end': end,
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest()}


def select_source_tokens(span, offsets, author_keep, gold):
    """Use offsets of ' '+prompt and preserve every original eligible gold token."""
    if any(len(pair) != 2 or pair[0] < 0 or pair[1] < pair[0] for pair in offsets):
        raise ValueError('Invalid token offsets')
    keep = sorted(set(map(int, author_keep)))
    if not keep or keep[0] < 0 or keep[-1] >= len(offsets):
        raise ValueError('Invalid author keep indices')
    start, end = span['start'] + 1, span['end'] + 1
    # Full containment ensures deleting a source token cannot edit a question or
    # demonstration character even when a tokenizer merges a boundary token.
    selected = [i for i in keep if start <= offsets[i][0] < offsets[i][1] <= end]
    if not selected:
        raise ValueError('No eligible evidence-body tokens')
    original_gold = set(map(int, gold)) & set(keep)
    if not original_gold:
        raise ValueError('source-v2 requires nonempty eligible recovery gold')
    lost = original_gold - set(selected)
    if lost:
        raise ValueError(f'Source parser would discard {len(lost)} eligible gold tokens; refusing to shrink the denominator')
    return selected


def recovery_curve(scores, keep, gold, fractions):
    """Deterministic token-budget curve, identical for DT and live FT."""
    scores = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    points = []
    for fraction in fractions:
        d = recovery_diagnostics(scores, keep, gold, fraction)
        d.pop('selected')
        chance, ceiling = d['random_expected_recall'], d['ceiling']
        d['fraction'] = float(fraction)
        d['chance_adjusted_recall'] = (d['recall'] - chance) / (ceiling - chance) if ceiling > chance else None
        points.append(d)
    return {'budget_unit': 'source_eligible_tokens', 'score_view': 'positive_part',
            'tie_break': 'ascending_prompt_token_index', 'points': points}


def reference_token_ids(input_ids, eligible_positions, eos_token_id):
    """Construct the exact intervention shared with deletion's final endpoint."""
    result = list(map(int, input_ids))
    for position in eligible_positions:
        if not 0 <= position < len(result):
            raise ValueError('Reference position is outside the input')
        result[position] = int(eos_token_id)
    return result


def sentence_recovery_curve(prompt, span, offsets, scores, keep, gold, fractions):
    """Optional whole-unit retrieval; sentence budgets are not token budgets.

    Sentence/line units use a fixed punctuation/newline splitter. Every token in
    a selected unit is retrieved. Gold only labels precomputed units afterwards.
    """
    body = prompt[span['start']:span['end']]
    cuts = [span['start'] + m.end() + 1 for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+', body)]
    groups = {}
    for i in sorted(keep):
        group = int(np.searchsorted(cuts, offsets[i][0], side='right'))
        groups.setdefault(group, []).append(i)
    units = list(groups.values())
    positive = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    values = np.asarray([positive[tokens].mean() for tokens in units], dtype=np.float32)
    gold = set(gold) & set(keep)
    gold_units = [i for i, tokens in enumerate(units) if gold & set(tokens)]
    points = []
    for fraction in fractions:
        d = recovery_diagnostics(values, range(len(units)), gold_units, fraction)
        selected = d.pop('selected')
        d['fraction'] = float(fraction)
        d['selected_token_count'] = sum(len(units[i]) for i in selected)
        d['selected_token_fraction'] = d['selected_token_count'] / len(keep)
        points.append(d)
    return {'budget_unit': 'sentence_or_line_units', 'selection': 'whole_units',
            'score_view': 'mean_positive_source_token_score',
            'splitter': 'newline_or_sentence_punctuation_followed_by_whitespace_v1',
            'points': points}
