"""Group a controlled source-key V-reference diagnostic, without credit correction."""
import hashlib
import json
from pathlib import Path
import time

from analyze_attention_gate import state_frequency
from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def summary(points):
    result = dict(positions=len(points),states=len({p['initial_state_sha256'] for p in points}),
                  trajectories=len({p['traj_uid'] for p in points}))
    if not points:
        return dict(**result,missing='No measured positions, not zero')
    improved = lambda p:abs(p['controlled_core_residual']) < abs(p['baseline_core_residual'])
    result.update(
        absolute_local_residual_reduced_positions=sum(improved(p) for p in points),
        state_equal_reduced_frequency=state_frequency(points,improved),
        baseline_signed_residual=quantiles([p['baseline_core_residual'] for p in points]),
        controlled_signed_residual=quantiles([p['controlled_core_residual'] for p in points]),
        baseline_absolute_residual=quantiles([abs(p['baseline_core_residual']) for p in points]),
        controlled_absolute_residual=quantiles([abs(p['controlled_core_residual']) for p in points]),
        paired_absolute_residual_change=quantiles([abs(p['controlled_core_residual'])-abs(p['baseline_core_residual']) for p in points]),
        baseline_minus_historical_residual=quantiles([p['baseline_core_residual']-p['historical_core_residual'] for p in points]),
        unchanged_value_route_maxabs=max(abs(p['value_coefficient_contraction_difference']) for p in points))
    return result


def main():
    local = HERE/'FA-source-reference-v1'
    raw_path = local/'result.json'
    raw = json.loads(raw_path.read_bytes())
    protocol_path = local/'protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    assert raw['phase'] == 'complete' and len(raw['points']) == 165
    frozen = {(p['traj_uid'],p['packed_slot']):p for p in protocol['points']}
    prior_path = HERE/'single-background-textcraft-analysis.json'
    prior = {(p['traj_uid'],p['packed_slot']):p for p in json.loads(prior_path.read_bytes())['points']}
    assert set(frozen) == set(prior) == {(p['traj_uid'],p['packed_slot']) for p in raw['points']}
    points = raw['points']
    for p in points:
        key = p['traj_uid'],p['packed_slot']
        source, reference = frozen[key],prior[key]
        assert p['token_id'] == source['token_id'] == reference['token_id']
        p['cohorts'] = source['cohorts']
        p['prediction_bin'] = stratum(source['saved_d'])
        p['reference_possible_bins'] = sorted({stratum(d) for d in reference['native_interval']})
        p['robust_spurious_global_tail'] = reference['original_spurious_tail']
        # Keep unresolved numerical bins explicit; neither select a reference
        # nor treat the oracle operand contraction as a new global d estimate.
    cohorts = {}
    for cohort in ('uniform','predicted_tail_census'):
        rows = [p for p in points if cohort in p['cohorts']]
        cells = {}
        for p in rows:
            cells.setdefault(p['prediction_bin']+':'+','.join(p['reference_possible_bins']),[]).append(p)
        cohorts[cohort] = dict(all=summary(rows),
            crossed_prediction_reference_cells={k:summary(v) for k,v in cells.items()},
            robust_spurious_tail=summary([p for p in rows if p['robust_spurious_global_tail']]),
            by_initial_state={s:summary([p for p in rows if p['initial_state_sha256']==s])
                for s in sorted({p['initial_state_sha256'] for p in rows})},
            by_exposure={str(e):summary([p for p in rows if p['previously_examined']==e]) for e in (False,True)})
    result = dict(scope=__doc__,unix=time.time(),sources=[ref(raw_path),ref(protocol_path),ref(prior_path),ref(Path(__file__))],
        task='textcraft',cohorts=cohorts,points=points,
        runtime={k:raw[k] for k in ('pid','birth','elapsed_seconds','operations','sampled_PSS_peak_bytes',
            'GPU_peak_allocated_bytes','GPU_peak_reserved_bytes')},
        baseline_replay=raw['batches'],production_modified=False,credit_repaired=False,
        official_tolerance_test=False,
        interpretation='Paired local operator residuals under one source-key V oracle intervention. '
            'The original collection is not the unified probability sample and does not supply a recall estimate. '
            'No global d, A, policy gradient or RISE/MAS improvement is computed here. '
            'Uniform and tail-census sources, crossed numerical bins and states remain separate. '
            'This oracle is unavailable to a one-DT training rule; it must not be deployed as token refinement.')
    output = REPO/'experiments/rl/results_saved_FA_source_reference_20261009.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output),runtime=result['runtime'],
        cohorts={c:dict(all=r['all'],robust_spurious_tail=r['robust_spurious_tail']) for c,r in cohorts.items()})))


if __name__=='__main__':
    main()
