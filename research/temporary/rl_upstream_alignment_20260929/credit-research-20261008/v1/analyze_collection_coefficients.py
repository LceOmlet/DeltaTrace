"""Stratified coefficient errors and likelihood feasibility; no new model query."""
import hashlib
import json
from pathlib import Path

from summarize_author_collection import stratum, quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def receipt(path):
    return dict(path=path.resolve().as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def bound_summary(points):
    rows = [dict(p, predicted_band=stratum(p.get('d', p.get('saved_d'))),
                 native_band=stratum(p['native_single_d']),
                 implied_deleted_logp=p['factual_target_logp'] - p.get('d', p.get('saved_d')))
            for p in points]
    positive = [p for p in rows if p['implied_deleted_logp'] > 0]
    cells = {}
    for key in sorted({(p['predicted_band'], p['native_band']) for p in rows}):
        cell = [p for p in rows if (p['predicted_band'], p['native_band']) == key]
        gaps = [p for p in cell if p['implied_deleted_logp'] > 0]
        cells[' -> '.join(key)] = dict(points=len(cell), positive_implied_deleted_logp=len(gaps),
            positive_margin_quantiles=quantiles([p['implied_deleted_logp'] for p in gaps]),
            by_state={state: dict(points=sum(p['initial_state_sha256'] == state for p in cell),
                positive_implied_deleted_logp=sum(p['initial_state_sha256'] == state for p in gaps))
                for state in sorted({p['initial_state_sha256'] for p in cell})})
    return dict(points=len(rows), positive_implied_deleted_logp=len(positive),
                positive_states=len({p['initial_state_sha256'] for p in positive}),
                cells=cells, positive_points=positive)


def main():
    coefficient_path = HERE / 'collection-coefficients.json'
    coefficient = json.loads(coefficient_path.read_bytes())
    resource = json.loads((HERE / 'collection-coefficient-resource.json').read_bytes())
    uniform = dict(textcraft=[], appworld=[])
    tails = dict(textcraft=[], appworld=[])
    sources = []
    for rank in (0, 1):
        path = HERE / f'author-collection-observations/rank{rank}.json'
        sources.append(receipt(path))
        observed = json.loads(path.read_bytes())
        for batch in observed['batches']:
            for trajectory in batch['trajectories']:
                uniform[batch['task']].extend(dict(p, traj_uid=trajectory['traj_uid'],
                    initial_state_sha256=trajectory['initial_state_sha256'])
                    for p in trajectory['uniform_deletions'])
        tails['textcraft' if rank == 0 else 'appworld'] = observed['tail_results']
    result = dict(scope=__doc__, status='Completed CPU-only original-artifact analysis; no training change',
        coefficient_receipt=receipt(coefficient_path), coefficient=coefficient,
        frozen_manifest=receipt(HERE / 'manifest.json'), native_sources=sources,
        physical_bound=dict(derivation='If d is a single-EOS log probability difference, inferred log p_deleted(Y)=log p_factual(Y)-d must be <=0. Each selected target conditional log probability is nonpositive, so their joint sum has this necessary bound.',
            interpretation='Positive margins below are diagnostics under measured factual scores. They are not new FA/FLA tolerances, a world-counterfactual oracle, or a clipping/correction rule. Tiny margins need their existing numerical reference before a numerical verdict; all margins remain visible.',
            uniform={task:bound_summary(points) for task,points in uniform.items()},
            predicted_tail_census={task:bound_summary(points) for task,points in tails.items()}),
        resource_observation=resource,
        conclusions=[
            'All 37 old source bindings match the earlier receipt exactly; 128 fixed uniform source bindings also resolve original token IDs, original source masks and retained training slots.',
            'The original FP32 reward composition reproduces all 165 saved raw source advantages with observed maximum absolute difference 0; this is a composition/interface check, not evidence of attribution accuracy.',
            'Raw-credit sign errors and fixed-original-moment actor-coefficient sign errors differ. Neither count is a gradient-impact estimate.',
            'Keep uniformly sampled bounded cells, native tails missed by DT, and the full predicted-tail census separate. Do not extrapolate unweighted sampled coefficient sums into a full-bulk gradient.',
            'The next unresolved measurement is the native-loss gradient of coefficient differences, not another measurement of current tail contributions. The existing optional PG observer provides this seam without a new PPO implementation.',
            'No GDN candidate is justified by this coefficient-level evidence alone. Formal jobs remain held; no corrected credits or checkpoints are deployed.'
        ],
        script=receipt(Path(__file__)))
    for task in uniform:
        assert len(uniform[task]) == 128 and len(tails[task]) == 37
    assert coefficient['original_tail_mapping_exact']
    assert max(abs(p['CPU_original_composition_difference']) for p in coefficient['records']) == 0
    target = REPO / 'experiments/rl/results_credit_coefficient_errors_20261008.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(target), physical_bound={cohort:{task:{k:v for k,v in spec.items()
        if k in ('points','positive_implied_deleted_logp','positive_states')}
        for task,spec in result['physical_bound'][cohort].items()}
        for cohort in ('uniform','predicted_tail_census')})))


if __name__ == '__main__':
    main()
