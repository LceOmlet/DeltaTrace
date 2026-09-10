"""Verify reserved task-specific target validation without candidate selection."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from evidence_protocol import source_span, select_source_tokens, reference_token_ids, recovery_curve
from retrieval_views import sentence_density_order
from analyze import paired_bootstrap, interval
from analyze_recall_pilot import same_tree

TASKS = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
FRACTIONS = [.05, .1, .2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_records(records, vectors, caches, originals, tok):
    rows, diagnostics, residuals = [], [], []
    identity_checks = 0
    for row in records:
        t, i, mode = row['dataset'], row['index'], row['target_mode']
        cache, old = caches[t][i], originals[t][i]
        assert row['status'] == 'complete'
        assert row['original_target_sha256'] == hashlib.sha256(cache['target'].encode()).hexdigest()
        original_target = tok.encode(cache['target'], add_special_tokens=False)
        start, end = cache['indices_to_explain']
        assert row['original_answer_token_span'] == [start, end]
        cs, ce = original_target.offsets[start][0], original_target.offsets[end][1]
        assert row['original_answer_char_span'] == [cs, ce]
        expected_target = cache['target'][cs:ce] if mode == 'answer_only' else cache['target']
        assert row['target'] == expected_target
        ids = np.asarray(row['input_ids'], dtype=np.int64)
        assert hashlib.sha256(ids.tobytes()).hexdigest() == row['input_sha256']
        positions = row['user_positions']
        for name in ('user_positions', 'gold', 'prompt_length'):
            assert row[name] == old[name]
        assert row['author_keep'] == old['keep']
        assert row['input_ids'][:row['prompt_length']] == old['input_ids'][:old['prompt_length']]
        eos = int(old['input_ids'][-1])
        target_ids = tok.encode(expected_target, add_special_tokens=False).ids + [eos]
        assert row['input_ids'][row['prompt_length']:] == target_ids
        assert row['target_length'] == len(target_ids)
        text = ' ' + cache['prompt']
        enc = tok.encode(text, add_special_tokens=False)
        assert len(enc.ids) == len(positions)
        assert np.flatnonzero(np.asarray(enc.ids) != ids[positions]).tolist() == row['standalone_token_boundary_differences']
        span = source_span(t, cache['prompt'])
        assert span == row['source_span']
        keep = select_source_tokens(span, enc.offsets, row['author_keep'], row['gold'])
        assert keep == row['keep']
        full_ref = np.asarray(reference_token_ids(ids, [positions[j] for j in row['author_keep']], eos), dtype=np.int64)
        assert hashlib.sha256(full_ref.tobytes()).hexdigest() == row['references']['full']
        weights = np.asarray(row['target_weights'], dtype=np.float64)
        assert weights.shape == (len(target_ids),) and set(weights) <= {0., 1.} and weights.sum() > 0
        for hops in (1, 3):
            first = row[f'FT_K{hops}_actual_target_aggregation'][0]
            actual = np.zeros_like(weights)
            lo, hi = first['start'], first['end']
            assert 0 <= lo <= hi < len(weights) - 1
            actual[lo:hi+1] = first['weights'] if first['weights'] is not None else 1.
            assert np.array_equal(actual, weights)
            assert (lo, hi) == (0, len(target_ids)-2)
            assert all((item['start'],item['end'])==(0,len(target_ids)-2) for item in row[f'FT_K{hops}_actual_target_aggregation'])
        prefix = f'{t}_{i}_{mode}_'
        if row.get('weighted_identity_bitwise_equal'):
            assert mode == 'full'
            assert np.array_equal(vectors[prefix+'DT_original_signed_full'], vectors[prefix+'DT_weighted_identity_signed_full'])
            identity_checks += 1
        for method, views in row['metrics'].items():
            if method.startswith('DT_'):
                signed = vectors[prefix + method + '_signed_full']
                assert signed.shape == ids.shape and np.isfinite(signed).all()
                values = signed[positions].astype(np.float32)
                detail = row[method + '_details']
                l0, l1 = np.asarray(detail['endpoint_target_logprobs32'], dtype=np.float64)
                w = weights if method == 'DT_target' else np.ones_like(weights)
                delta = float(((l1 - l0) * w).sum())
                assert np.isclose(delta, detail['target_delta_score32_sum64'], atol=1e-8)
                assert np.isclose(signed.sum(), detail['signed_sum'], atol=1e-10)
                residuals.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method,
                    'target_delta': delta, 'signed_sum': float(signed.sum()),
                    'residual': detail['unassigned_total']})
            else:
                assert method in ('FT_K1', 'FT_K3')
                values = vectors[prefix + method + '_prompt'].astype(np.float32)
            positive = np.maximum(values, 0)
            order = sentence_density_order(text, enc.offsets, positive, keep)
            assert len(order) == len(keep) and set(order) == set(keep)
            rank = np.zeros_like(positive)
            rank[order] = np.arange(len(keep), 0, -1)
            for view, scores in [('raw', positive), ('density', rank)]:
                curve = recovery_curve(scores, keep, row['gold'], FRACTIONS)
                same_tree(views[view], curve)
                for point in curve['points']:
                    rows.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method, 'view': view,
                        'fraction': point['fraction'], 'recall': point['recall'], 'budget': point['budget'],
                        'gold': point['gold'], 'ceiling': point['ceiling']})
            if t.startswith('vt_') and method in ('DT_original', 'DT_target', 'FT_K3'):
                assignments = [(m.start(), m.end()) for m in re.finditer(r'\bVAR [A-Z]+\s*=\s*(?:VAR [A-Z]+|\d+)', text)
                    if span['start']+1 <= m.start() < m.end() <= span['end']+1]
                assignment_tokens = {j for j in keep if any(enc.offsets[j][0] < b and enc.offsets[j][1] > a for a, b in assignments)}
                gold = set(row['gold']) & set(keep)
                mass = positive[list(assignment_tokens)].sum()
                k = int(np.ceil(.1 * len(keep)))
                selected = set(order[:k])
                diagnostics.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method,
                    'gold_share_of_assignment_positive_mass': float(positive[list(gold & assignment_tokens)].sum()/mass) if mass > 0 else None,
                    'distractor_assignment_tokens_in_budget': len(selected & (assignment_tokens - gold)),
                    'assignment_count': len(assignments), 'budget': k})
    return rows, diagnostics, residuals, identity_checks
