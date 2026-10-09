"""Read-only bounded-cell credit error and probability-bound diagnostics.

Keep the frozen probability sample and every saved numerical reference. This
does not normalize, clip, repair, or replace any production credit.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_tail_probability_sample import weighted_quantiles, ratio_bin
from summarize_author_collection import quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LOCAL = HERE/'tail-probability-sample-v1'


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def summarize(points, *, distributions=True):
    base = dict(observed_positions=len(points),observed_states=len({p['initial_state_sha256'] for p in points}))
    if not points:
        return dict(**base,missing='No observations, not zero')
    w = [p['token_total_weight'] for p in points]
    base.update(HT_source_total=math.fsum(w),
        probability_bound=dict(
            violated_all_recorded_factual_views=sum(p['implied_deleted_logp_interval'][0]>0 for p in points),
            crossed_zero=sum(p['implied_deleted_logp_interval'][0]<=0<p['implied_deleted_logp_interval'][1] for p in points),
            not_violated_any_recorded_view=sum(p['implied_deleted_logp_interval'][1]<=0 for p in points)),
        DT_d_below_all_native_views=sum(p['d_error_interval'][1]<0 for p in points),
        DT_d_above_all_native_views=sum(p['d_error_interval'][0]>0 for p in points),
        DT_d_within_native_range=sum(p['d_error_interval'][0]<=0<=p['d_error_interval'][1] for p in points))
    if not distributions:
        return base
    base.update(
        d_error_low_weighted_quantiles=weighted_quantiles([p['d_error_interval'][0] for p in points],w),
        d_error_high_weighted_quantiles=weighted_quantiles([p['d_error_interval'][1] for p in points],w),
        A_over_r_error_low_weighted_quantiles=weighted_quantiles([p['A_over_r_error_interval'][0] for p in points],w),
        A_over_r_error_high_weighted_quantiles=weighted_quantiles([p['A_over_r_error_interval'][1] for p in points],w))
    return base


def main():
    start = time.perf_counter()
    prior_path = REPO/'experiments/rl/results_stable_negative_credit_20261008.json'
    prior = json.loads(prior_path.read_bytes())['tasks']
    result = dict(scope=__doc__,unix=time.time(),sources=[ref(Path(__file__)),ref(prior_path)],tasks={},
        operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0),production_modified=False,
        probability_boundary='For a normalized joint text target p_deleted(Y)<=1, so d>=log p_factual(Y). '
            'An implied log p_deleted>0 violates this bound relative to that factual view. '
            'A range of recorded views is not a rigorous oracle error bound.',
        credit_error='epsilon=d_DT-d_native; A_DT/r-A_native/r=exp(-d_native)-exp(-d_DT). '
            'Reported as intervals over the saved native views, without selecting a favorable reference.',
        aggregation='Only within predicted/native magnitude cells; no pooled raw advantage moment, '
            'finite-variance claim, new pass threshold, or inferred policy-gradient fraction.',
        sampling='Exactly the prior frozen 1061/task probability sample. Earlier records only widen reference ranges '
            'at matching identities; no earlier query is appended and inclusion probabilities remain unchanged.')
    for task in ('textcraft','appworld'):
        path = LOCAL/'analysis-c9e78c9b'/f'{task}.json'
        data = json.loads(path.read_bytes())['tasks'][task]
        previous = {(p['traj_uid'],p['packed_slot']):p for c in prior[task]['cohorts'].values()
                    for p in c['points_with_identity']}
        points = []
        for p in data['observations']:
            key = p['traj_uid'],p['packed_slot']
            old = previous.get(key)
            factual = [view['factual'] for view in p['scores'].values()]
            if old is not None:
                assert old['token_id'] == p['token_id']
                factual.extend(old['factual_logp_observed_interval'])
            lo,hi = p['native_d_interval']
            d = p['saved_d']
            p = dict(p,
                factual_logp_observed_interval=[min(factual),max(factual)],
                implied_deleted_logp_interval=[min(factual)-d,max(factual)-d],
                d_error_interval=[d-hi,d-lo],
                A_over_r_error_interval=[math.exp(-hi)-math.exp(-d),math.exp(-lo)-math.exp(-d)],
                reference_bin_interval=[ratio_bin(hi),ratio_bin(lo)])
            assert all(math.isfinite(x) for x in p['A_over_r_error_interval'])
            points.append(p)
        assert len(points) == 1061
        cells = {}
        for p in points:
            label=ratio_bin(p['saved_d'])+':reference_'+ '..'.join(p['reference_bin_interval'])
            cells.setdefault(label,[]).append(p)
        predicted = {s['name']:summarize([p for p in points if ratio_bin(p['saved_d'])==s['name']],distributions=False)
            for s in data['frame']['strata']}
        assert sum(g['observed_positions'] for g in predicted.values()) == len(points)
        assert all(p['prediction_stratum']==ratio_bin(p['saved_d']) for p in points)
        result['tasks'][task] = dict(frame=data['frame'],observations=points,
            crossed_prediction_reference_cells={k:summarize(v) for k,v in cells.items()},
            probability_boundary_by_prediction=predicted,
            probability_boundary_by_state={s:{label:summarize([p for p in points
                    if p['initial_state_sha256']==s and ratio_bin(p['saved_d'])==label],distributions=False)
                for label in predicted} for s in sorted({p['initial_state_sha256'] for p in points})})
        result['sources'].append(ref(path))
    # Separate earlier operator evidence from the new probability frame. This
    # measures terms of a recorded decomposition, not causal error shares.
    op_path = HERE/'attention-pv-textcraft-v2/analysis.json'
    operators = json.loads(op_path.read_bytes())['points']
    result['sources'].append(ref(op_path))
    result['earlier_operator_collection_not_recall_frame'] = {}
    for cohort in ('uniform','predicted_tail_census'):
        rows = [p for p in operators if cohort in p['cohorts']]
        names = ('joint_QK_softmax_residual','PV_background_residual','native_pair_core_residual',
                 'matched_minus_native_endpoint_residual')
        result['earlier_operator_collection_not_recall_frame'][cohort] = dict(
            positions=len(rows),states=len({p['initial_state_sha256'] for p in rows}),
            FA_terms={n:dict(signed=quantiles([p['final_FA_PV_ledger'][n] for p in rows]),
                absolute=quantiles([abs(p['final_FA_PV_ledger'][n]) for p in rows])) for n in names},
            head_logsoftmax_background_absolute=quantiles([
                abs(p['suboperations']['head']['native_terms']['logsoftmax_background']) for p in rows]),
            interpretation='Recorded local decomposition terms can cancel. Magnitudes are not a fraction of global credit error. '
                'No old point is added to the new sampling frame or its confusion matrix.')
    result['elapsed_seconds'] = time.perf_counter()-start
    path = REPO/'experiments/rl/results_probability_sample_credit_error_20261009.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(path),elapsed=result['elapsed_seconds'],
        tail_boundary={t:{k:v for k,v in d['probability_boundary_by_prediction'].items()
            if k in ('ratio_2_to_10','ratio_10_to_100','ratio_gt_100')} for t,d in result['tasks'].items()})))


if __name__=='__main__':
    main()
