"""CPU census of saved formal credit against the necessary probability bound.

For a normalized target, interpreting d as log(p_factual/p_deleted) requires
log(p_factual)-d <= 0. This is a necessary check, not a single-deletion oracle,
precision/recall estimate, kernel tolerance or training rule. Retain all saved
DP duplicate views rather than picking whichever numerical view is favorable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from summarize_tail_probability_sample import weighted_quantiles

NAMES = ('ratio_le_1', 'ratio_1_to_2', 'ratio_2_to_10', 'ratio_10_to_100', 'ratio_gt_100')


def binding(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def census(audit_path):
    audit = json.loads(audit_path.read_bytes())
    paths = sorted({row['path'] for row in audit['rows']})
    groups = {}
    for path in paths:
        for row in torch.load(path, map_location='cpu', weights_only=False)['rows']:
            groups.setdefault(row['traj_uid'], []).append(row)
    trajectories, by_state = [], {}
    thresholds = torch.tensor([0., math.log(2), math.log(10), math.log(100)], dtype=torch.float64)
    for uid, views in groups.items():
        row = views[0]
        for other in views[1:]:
            for name in ('input_ids', 'suffix_positions', 'policy_mask', 'target_mask', 'prior_source_mask'):
                assert torch.equal(row[name], other[name]), (uid, name)
        indices = row['prior_source_mask'].nonzero().flatten()
        offsets = torch.searchsorted(row['suffix_positions'], indices)
        assert torch.equal(row['suffix_positions'][offsets], indices)
        positions = row['prompt_length']+offsets
        d = torch.stack([view['source_log_ratios'][indices].double() for view in views])
        factual = torch.tensor([view['factual_target_logp'] for view in views], dtype=torch.float64)
        assert torch.isfinite(d).all() and torch.isfinite(factual).all()
        implied = factual[:, None]-d
        lo, hi = implied.amin(0), implied.amax(0)
        bins = torch.bucketize(-d, thresholds)
        bin_lo, bin_hi = bins.amin(0), bins.amax(0)
        first = int(row['policy_mask'].nonzero().flatten()[0])
        first_offset = int((row['suffix_positions'] == first).nonzero().flatten()[0])
        state = hashlib.sha256(row['input_ids'][:row['prompt_length']+first_offset].numpy().tobytes()).hexdigest()
        output = dict(traj_uid=uid, initial_state_sha256=state, DP_views=len(views),
            input_length=row['input_ids'].numel(), prior_source_tokens=indices.numel(),
            factual_target_logp_range=[float(factual.min()), float(factual.max())],
            bound_violations_all_views=int((lo > 0).sum()),
            bound_violations_any_view=int((hi > 0).sum()),
            groups={})
        for i, name in enumerate(NAMES):
            mask = (bin_lo == i) & (bin_hi == i)
            output['groups'][name] = dict(tokens=int(mask.sum()),
                violations_all_views=int(((lo > 0) & mask).sum()),
                violations_any_view=int(((hi > 0) & mask).sum()),
                implied_deleted_logp_low_quantiles=weighted_quantiles(lo[mask].tolist(), [1]*int(mask.sum())),
                implied_deleted_logp_high_quantiles=weighted_quantiles(hi[mask].tolist(), [1]*int(mask.sum())))
        output['DP_views_cross_ratio_bins'] = int((bin_lo != bin_hi).sum())
        violating = (hi > 0).nonzero().flatten()
        output['violating_positions'] = [dict(response_position=int(indices[j]), input_position=int(positions[j]),
            token_id=int(row['input_ids'][positions[j]]), d_range=[float(d[:, j].min()), float(d[:, j].max())],
            implied_deleted_logp_range=[float(lo[j]), float(hi[j])]) for j in violating]
        trajectories.append(output)
        by_state.setdefault(state, []).append(output)
    def counts(rows):
        return dict(trajectories=len(rows), source_tokens=sum(r['prior_source_tokens'] for r in rows),
            violations_all_views=sum(r['bound_violations_all_views'] for r in rows),
            violations_any_view=sum(r['bound_violations_any_view'] for r in rows),
            trajectories_with_all_view_violations=sum(r['bound_violations_all_views'] > 0 for r in rows),
            DP_views_cross_ratio_bins=sum(r['DP_views_cross_ratio_bins'] for r in rows),
            ratio_groups={name:{key:sum(r['groups'][name][key] for r in rows)
                for key in ('tokens', 'violations_all_views', 'violations_any_view')} for name in NAMES})
    return dict(audit=binding(audit_path), sources=[binding(p) for p in paths],
        original_rank_rows=sum(len(v) for v in groups.values()), unique_trajectories=len(groups),
        census=counts(trajectories), by_initial_state={state:counts(rows) for state,rows in by_state.items()},
        trajectories=trajectories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior diagnostic results'
    begin = time.perf_counter()
    result = dict(scope=__doc__, cohorts={str(p):census(p) for p in args.audit},
        source=binding(__file__), grouping='Existing PLAN ratio bins; separate completed formal iterations and initial states.',
        denominator='Census of saved nonzero DT requests only, unique UID/source positions; no zero-reward or missing first iteration is filled.',
        duplicate_views='All saved original DP padding views retained as numerical sensitivity, not independent trajectories.',
        numerical_scope='Strict inequalities relative to recorded factual log-prob values; no new numerical tolerance or confidence interval.',
        limitation='Within-bound does not prove accurate credit. No native deletion query, recall estimate, correction, clipping or deployment.',
        operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0))
    assert not torch.cuda.is_initialized()
    result.update(seconds=time.perf_counter()-begin, unix=time.time(), CUDA_initialized=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'], output=binding(args.output),
        cohorts={name:value['census'] for name,value in result['cohorts'].items()})))
