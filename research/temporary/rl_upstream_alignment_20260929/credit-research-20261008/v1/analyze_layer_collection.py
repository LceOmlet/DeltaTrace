"""Describe passive layer residuals without pooling bulk and tail means."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_author_collection import stratum, quantiles

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def describe(points):
    if not points:
        return dict(points=0, trajectories=0, states=0,
                    status='Structurally empty cell; no numerical values imputed')
    states = {}
    for p in points:
        states.setdefault(p['initial_state_sha256'], []).append(p)
    def layer_counts(items):
        counts = [0]*33
        for p in items:
            if any(p['residuals']):
                counts[max(range(33), key=lambda i: abs(p['residuals'][i]))] += 1
        return counts
    state_rows = [dict(initial_state_sha256=state, points=len(items),
        largest_abs_residual_layer_counts=layer_counts(items),
        residuals_by_layer={str(i):dict(signed=quantiles([p['residuals'][i] for p in items]),
            absolute=quantiles([abs(p['residuals'][i]) for p in items])) for i in range(33)})
        for state, items in sorted(states.items())]
    return dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
        states=len(states), state_groups=state_rows,
        exactly_zero_residual_points=sum(not any(p['residuals']) for p in points),
        largest_abs_residual_layer_counts=layer_counts(points),
        state_equal_largest_abs_residual_layer_frequencies=[sum(
            row['largest_abs_residual_layer_counts'][i]/row['points'] for row in state_rows)/len(state_rows)
            if state_rows else None for i in range(33)],
        residuals_by_layer={str(i):dict(signed=quantiles([p['residuals'][i] for p in points]),
            absolute=quantiles([abs(p['residuals'][i]) for p in points])) for i in range(33)},
        original_abs_d_error=quantiles([abs(p['comparison_DT_d']-p['comparison_native_d']) for p in points]),
        fresh_abs_d_error=quantiles([abs(p['fresh_DT_d']-p['native_single_d']) for p in points]),
        fresh_DT_minus_s0=quantiles([p['fresh_DT_d']-p['boundaries']['0']['value'] for p in points]),
        DT_batching_drift=quantiles([p['fresh_DT_d']-p['comparison_DT_d'] for p in points]),
        native_batching_drift=quantiles([p['native_single_d']-p['comparison_native_d'] for p in points]),
        factual_score_native_minus_DT=quantiles([p['factual_target_logp']-p['DT_factual_target_logp']
            for p in points if p['DT_factual_target_logp'] is not None]),
        factual_DT_score_missing=sum(p['DT_factual_target_logp'] is None for p in points),
        telescoping_roundoff=quantiles([p['telescoping_roundoff'] for p in points]),
        factual_endpoint_checks={str(i):dict(nonidentical=sum(
            not p['boundaries'][str(i)]['factual_endpoints_equal'] for p in points),
            maxabs=quantiles([p['boundaries'][str(i)]['factual_endpoint_maxabs'] for p in points])) for i in range(33)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('textcraft', 'appworld'), required=True)
    args = parser.parse_args()
    task = args.task
    folder = HERE/('layer-'+task+'-observations')
    rank_paths = [folder/f'rank{rank}.json' for rank in (0, 1)]
    ranks = [json.loads(p.read_bytes()) for p in rank_paths if p.exists()]
    points = []
    for rank in ranks:
        for batch in rank['batches']:
            for p in batch['points']:
                slot = batch['uids'].index(p['traj_uid'])
                audits = batch['DT_detail'].get('per_sample', [])
                score = audits[slot].get('factual_target_logp') if slot < len(audits) else None
                points.append(dict(p, DT_factual_target_logp=score))
    assert len({(p['traj_uid'], p['packed_slot']) for p in points}) == len(points)
    output = {}
    for cohort in ('uniform', 'predicted_tail_census'):
        selected = []
        for p in points:
            for comparison in p['previous_comparisons']:
                if comparison['cohort'] == cohort:
                    d = comparison.get('saved_d', comparison.get('d'))
                    selected.append(dict(p, comparison_DT_d=d,
                        comparison_native_d=comparison['native_single_d'],
                        crossed_stratum=stratum(d)+':'+stratum(comparison['native_single_d'])))
        cells = {}
        for cell in sorted({p['crossed_stratum'] for p in selected}):
            rows = [p for p in selected if p['crossed_stratum'] == cell]
            cells[cell] = dict(all=describe(rows), by_exposure={str(seen):describe([
                p for p in rows if p['previously_examined'] == seen]) for seen in (False, True)})
        output[cohort] = dict(measured_points=len(selected), crossed_ratio_cells=cells)
    result = dict(scope=__doc__, task=task, observations=[ref(p) for p in rank_paths if p.exists()],
        complete=len(ranks)==2 and all(r['phase']=='complete' for r in ranks),
        phase_by_rank={str(r['rank']):r['phase'] for r in ranks}, unique_points=len(points),
        cohorts=output, formal_release=False, new_GDN_candidate=False,
        interpretation='Layers 0..31 are whole decoder residuals; layer32 is final norm plus output head. Large opposing residuals can cancel, so the largest absolute term or its frequency is not a cause, a share of final error or an instruction to modify that layer. Ranking a whole decoder does not establish a GDN or kernel cause. Conditional distributions and bounded state-equal frequencies are localization diagnostics, not replacement quality metrics, official tolerance or training correction.')
    path = HERE/('layer-'+task+'-analysis.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(task=task, complete=result['complete'], unique_points=len(points),
        phase_by_rank=result['phase_by_rank'], cohorts={k:v['measured_points'] for k,v in output.items()})))


if __name__ == '__main__':
    main()
