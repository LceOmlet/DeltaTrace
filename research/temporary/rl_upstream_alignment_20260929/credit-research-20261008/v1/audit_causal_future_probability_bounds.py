"""CPU diagnostic of saved d using only targets causally after each source.

For an exact EOS single-delete log ratio, earlier target factors cancel.
Therefore sum(log p_factual(y_j), label_position_j > source_position_i)-d_i
must be <= 0. This strengthens the previous whole-joint necessary bound.
It is not an oracle, precision/recall estimate, official numerical tolerance,
training acceptance gate, clipping rule, or a replacement for native deletion.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import psutil
import torch

from audit_formal_credit_probability_bounds import NAMES, binding, weighted_quantiles


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
FORMAL = ROOT/'runs/textcraft-formal-stable-20261009-v1'


def quantiles(values):
    return weighted_quantiles(values.tolist(), [1]*values.numel())


def census(audit_path, packing):
    audit = json.loads(audit_path.read_bytes())
    paths = sorted({row['path'] for row in audit['rows']})
    views_by_uid = {}
    packing_residuals = []
    for path in paths:
        stored = torch.load(path, map_location='cpu', weights_only=True)
        batch = stored['rows']
        native = stored['native_target_diagnostics']
        # Use the actual owner's pre-suffix absolute positions and sample IDs.
        selected = packing.PackedAnswerTargets(
            [dict(prompt_length=r['prompt_length'], target_ids=r['input_ids'][r['prompt_length']:]) for r in batch],
            [r['target_offsets'] for r in batch], max(r['input_ids'].numel() for r in batch), 'cpu')
        factual = torch.as_tensor(native['target_logp1'], dtype=torch.float64)
        sums = selected.sample_sums(factual)
        assert torch.isfinite(factual).all() and (factual <= 0).all()
        for index, row in enumerate(batch):
            source = row['prior_source_mask'].nonzero().flatten()
            offsets = torch.searchsorted(row['suffix_positions'], source)
            assert torch.equal(row['suffix_positions'][offsets], source)
            positions = row['prompt_length']+offsets
            target_mask = selected.samples == index
            # The owner exposes causal predictor positions, one before labels.
            labels_at = selected.positions[target_mask]+1
            target_logp = factual[target_mask]
            assert torch.equal(row['input_ids'][labels_at], selected.labels[target_mask])
            # No source x target matrix: reverse cumsum then searchsorted.
            future_start = torch.searchsorted(labels_at, positions, right=True)
            reverse_sums = torch.cat((target_logp.flip(0).cumsum(0).flip(0), target_logp.new_zeros(1)))
            future_logp = reverse_sums[future_start]
            future_count = labels_at.numel()-future_start
            assert (future_count > 0).all()
            total = row['factual_target_logp']
            packing_residuals.append(float(sums[index])-total)
            views_by_uid.setdefault(row['traj_uid'], []).append(dict(
                path=path, row=row, source=source, positions=positions,
                future_logp=future_logp, future_count=future_count,
                total=total, d=row['source_log_ratios'][source].double()))
    thresholds = torch.tensor([0., math.log(2), math.log(10), math.log(100)], dtype=torch.float64)
    trajectories = []
    for uid, views in views_by_uid.items():
        first = views[0]
        row = first['row']
        for view in views[1:]:
            for name in ('input_ids', 'suffix_positions', 'policy_mask', 'target_mask', 'prior_source_mask'):
                assert torch.equal(row[name], view['row'][name]), (uid, name)
            assert torch.equal(first['future_count'], view['future_count'])
        d = torch.stack([v['d'] for v in views])
        future = torch.stack([v['future_logp'] for v in views])
        whole = torch.tensor([v['total'] for v in views], dtype=torch.float64)[:, None]
        assert torch.isfinite(d).all()
        excess = future-d
        whole_excess = whole-d
        lo, hi = excess.amin(0), excess.amax(0)
        whole_lo, whole_hi = whole_excess.amin(0), whole_excess.amax(0)
        bins = torch.bucketize(-d, thresholds)
        bin_lo, bin_hi = bins.amin(0), bins.amax(0)
        first_policy = int(row['policy_mask'].nonzero().flatten()[0])
        first_offset = int((row['suffix_positions'] == first_policy).nonzero().flatten()[0])
        state = hashlib.sha256(row['input_ids'][:row['prompt_length']+first_offset].numpy().tobytes()).hexdigest()
        output = dict(traj_uid=uid, initial_state_sha256=state, DP_views=len(views),
            source_tokens=len(first['source']), input_length=row['input_ids'].numel(),
            causal_violations_all_views=int((lo > 0).sum()), causal_violations_any_view=int((hi > 0).sum()),
            whole_violations_all_views=int((whole_lo > 0).sum()), whole_violations_any_view=int((whole_hi > 0).sum()),
            new_causal_violations_any_view=int(((hi > 0) & (whole_hi <= 0)).sum()),
            DP_views_cross_ratio_bins=int((bin_lo != bin_hi).sum()), groups={})
        for index, name in enumerate(NAMES):
            mask = (bin_lo == index) & (bin_hi == index)
            output['groups'][name] = dict(tokens=int(mask.sum()),
                causal_violations_all_views=int(((lo > 0) & mask).sum()),
                causal_violations_any_view=int(((hi > 0) & mask).sum()),
                whole_violations_any_view=int(((whole_hi > 0) & mask).sum()),
                positive_excess_high_quantiles=quantiles(hi[(hi > 0) & mask]))
        minimum = d.argmin().item()%d.shape[1]
        # Complete violation positions plus the minimum-d position, not a new sample.
        points = sorted(set((hi > 0).nonzero().flatten().tolist()+[minimum]))
        output['positions'] = [dict(
            response_position=int(first['source'][j]), input_position=int(first['positions'][j]),
            token_id=int(row['input_ids'][first['positions'][j]]), minimum_d_position=(j == minimum),
            future_target_tokens=int(first['future_count'][j]),
            views=[dict(path=v['path'], d=float(v['d'][j]),
                whole_factual_logp=v['total'], future_factual_logp=float(v['future_logp'][j]),
                implied_deleted_future_logp=float(v['future_logp'][j]-v['d'][j])) for v in views]) for j in points]
        trajectories.append(output)
    count_keys = ('source_tokens', 'causal_violations_all_views', 'causal_violations_any_view',
                  'whole_violations_all_views', 'whole_violations_any_view',
                  'new_causal_violations_any_view', 'DP_views_cross_ratio_bins')
    def counts(rows):
        return dict(trajectories=len(rows),
            trajectories_with_causal_violations=sum(r['causal_violations_any_view'] > 0 for r in rows),
            **{k:sum(r[k] for r in rows) for k in count_keys},
            ratio_groups={name:{k:sum(r['groups'][name][k] for r in rows) for k in
                ('tokens', 'causal_violations_all_views', 'causal_violations_any_view', 'whole_violations_any_view')}
                for name in NAMES})
    states = sorted({r['initial_state_sha256'] for r in trajectories})
    return dict(audit=binding(audit_path), sources=[binding(p) for p in paths],
        original_rank_rows=sum(len(v) for v in views_by_uid.values()),
        maximum_absolute_native_owner_sum_residual=max(map(abs, packing_residuals)),
        census=counts(trajectories), by_initial_state={s:counts([r for r in trajectories if r['initial_state_sha256']==s]) for s in states},
        minimum_raw_owner_row=min(audit['rows'], key=lambda r:r['minimum_prior_source']['raw_advantage']),
        trajectories=trajectories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous diagnostics'
    started = time.perf_counter()
    assert binding(FORMAL/'source.json')['sha256']=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
    manifest = json.loads((FORMAL/'source.json').read_bytes())
    sys.path.insert(0, str(Path(manifest['dt_root'])/'clean/qwen35'))
    packing = importlib.import_module('qwen35_answer_finite')
    assert binding(packing.__file__)['sha256']=='d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'
    result = dict(scope=__doc__,source=binding(__file__), packing_owner=binding(packing.__file__),
        formal_source=binding(FORMAL/'source.json'),cohorts={str(p):census(p,packing) for p in args.audit},
        numerical_scope='Strict positive excess relative to recorded FP32 log-probs, summed in FP64. No official whole-DT tolerance is invented.',
        population='Census of every source in the supplied complete nonzero-DT cohorts. Separate iterations, states and existing tail bins; DP views not independent.',
        limitation='Necessary autoregressive probability bound only. Passing does not establish single-delete accuracy; failing near zero can include numerical residuals. No precision/recall or world-oracle claim.',
        operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0),production_changes=0)
    assert not torch.cuda.is_initialized()
    result.update(seconds=time.perf_counter()-started,unix=time.time(),PSS_bytes=psutil.Process().memory_full_info().pss,CUDA_initialized=False)
    args.output.write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps(dict(output=binding(args.output),seconds=result['seconds'],PSS_bytes=result['PSS_bytes'],
        cohorts={k:v['census'] for k,v in result['cohorts'].items()})))


if __name__=='__main__':
    main()
