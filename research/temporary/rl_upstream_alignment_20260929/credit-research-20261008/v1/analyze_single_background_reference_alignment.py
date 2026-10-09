"""Join completed background diagnostics to recorded numerical references.

Retain capture/replay differences instead of silently using a fresh DT vector
as the original prediction. The old 165-point diagnostic collection is not the
1061-point probability sample and cannot supply a new method's recall.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_author_collection import quantiles
from summarize_tail_probability_sample import ratio_bin

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def tail_status(interval):
    low, high = interval
    boundary = -math.log(2)
    return 'tail' if high < boundary else 'not_tail' if low >= boundary else 'unresolved'


def summarize(rows):
    if not rows:
        return dict(positions=0, states=0, status='No observations; not zero error')
    components = ('capture_minus_diagnostic_joint', 'joint_minus_single_background',
                  'single_finite_minus_its_root', 'root_minus_native_low', 'root_minus_native_high')
    states = {}
    for p in rows:
        states.setdefault(p['initial_state_sha256'], []).append(p)
    return dict(positions=len(rows), states=len(states),
        trajectories=len({p['traj_uid'] for p in rows}),
        components={name:dict(signed=quantiles([p['components'][name] for p in rows]),
            absolute=quantiles([abs(p['components'][name]) for p in rows])) for name in components},
        capture_equals_diagnostic_joint=sum(p['components']['capture_minus_diagnostic_joint'] == 0 for p in rows),
        background_larger_than_other_terms=sum(p['background_larger_than_other_terms'] for p in rows),
        equal_state_frequency_background_larger=sum(
            sum(p['background_larger_than_other_terms'] for p in points)/len(points)
            for points in states.values())/len(states),
        by_state={key:dict(positions=len(points), trajectories=len({p['traj_uid'] for p in points}),
            background_larger_than_other_terms=sum(p['background_larger_than_other_terms'] for p in points))
            for key,points in states.items()})


def main():
    start = time.perf_counter()
    completed_path = REPO/'experiments/rl/results_single_background_completed_20261009.json'
    complete = json.loads(completed_path.read_bytes())
    assert complete['status'] == 'frozen_diagnostic_collection_complete_not_training_repair'
    sample_path = REPO/'experiments/rl/results_probability_sample_credit_error_20261009.json'
    probability = json.loads(sample_path.read_bytes())
    stable_path = REPO/'experiments/rl/results_stable_negative_credit_20261008.json'
    stable = json.loads(stable_path.read_bytes())
    result = dict(scope=__doc__, sources=[ref(__file__),ref(completed_path),ref(sample_path),ref(stable_path)],
        tasks={}, operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0,checkpoint_restore=0),
        production_modified=False, candidate=False,
        interpretation='Algebraic error ledger and paired diagnostic measurements only. No term is removed from credit, '
            'no new pass threshold is used, and a local term magnitude is not a causal share of global error. '
            'Reference ranges include every matching recorded view and are not rigorous oracle bounds. '
            'A single-background diagnostic with extra DT calls is not a cost-compliant training method.')
    for task in ('textcraft','appworld'):
        identity = complete['tasks'][task]['cross_cell_analysis']
        path = Path(identity['path'])
        assert ref(path) == identity
        single = json.loads(path.read_bytes())
        assert single['complete'] and len(single['points']) == 165
        old = {(p['traj_uid'],p['packed_slot']):p for cohort in stable['tasks'][task]['cohorts'].values()
            for p in cohort['points_with_identity']}
        current = {(p['traj_uid'],p['packed_slot']):p
            for p in probability['tasks'][task]['observations']}
        points=[]
        for p in single['points']:
            key = p['traj_uid'],p['packed_slot']
            original = old[key]
            assert original['token_id'] == p['token_id']
            low,high = p['native_interval']
            new = current.get(key)
            if new:
                assert new['token_id'] == p['token_id']
                low=min(low,new['native_d_interval'][0])
                high=max(high,new['native_d_interval'][1])
            captured_d = new['saved_d'] if new else original['DT_d']
            assert captured_d == original['DT_d']
            components=dict(capture_minus_diagnostic_joint=captured_d-p['joint_d'],
                joint_minus_single_background=p['joint_d']-p['single_d'],
                single_finite_minus_its_root=p['single_d']-p['single_root'],
                root_minus_native_low=p['single_root']-high,
                root_minus_native_high=p['single_root']-low)
            total_low=sum(components[n] for n in ('capture_minus_diagnostic_joint',
                'joint_minus_single_background','single_finite_minus_its_root','root_minus_native_low'))
            total_high=sum(components[n] for n in ('capture_minus_diagnostic_joint',
                'joint_minus_single_background','single_finite_minus_its_root','root_minus_native_high'))
            larger=abs(components['joint_minus_single_background'])>max(
                abs(components[n]) for n in ('capture_minus_diagnostic_joint',
                    'single_finite_minus_its_root','root_minus_native_low','root_minus_native_high'))
            points.append(dict(p,captured_d=captured_d,
                shared_with_new_probability_sample=new is not None,
                all_recorded_native_interval=[low,high],
                all_recorded_native_tail_status=tail_status([low,high]),
                components=components,background_larger_than_other_terms=larger,
                captured_prediction_bin=ratio_bin(captured_d),single_background_bin=ratio_bin(p['single_d']),
                reference_bin_interval=[ratio_bin(high),ratio_bin(low)],
                closure_roundoff=[total_low-(captured_d-high),total_high-(captured_d-low)]))
        cells={}
        for p in points:
            for cohort in p['cohorts']:
                label='|'.join([cohort,str(p['previously_examined']),p['captured_prediction_bin'],
                    '..'.join(p['reference_bin_interval'])])
                cells.setdefault(label,[]).append(p)
        tail=[p for p in points if 'predicted_tail_census' in p['cohorts']]
        assert len(tail) == 37 and all(p['shared_with_new_probability_sample'] for p in tail)
        categories={}
        for status in ('tail','not_tail','unresolved'):
            rows=[p for p in tail if p['all_recorded_native_tail_status']==status]
            categories[status]=dict(positions=len(rows),states=len({p['initial_state_sha256'] for p in rows}),
                single_diagnostic_still_tail=sum(p['single_d'] < -math.log(2) for p in rows),
                capture_equals_joint=sum(p['captured_d']==p['joint_d'] for p in rows))
        result['sources'].append(ref(path))
        result['tasks'][task]=dict(points=points,crossed_cells={k:summarize(v) for k,v in cells.items()},
            original_predicted_tail_census=categories,
            new_probability_sample_overlap=sum(p['shared_with_new_probability_sample'] for p in points),
            largest_ledger_roundoff=max(abs(v) for p in points for v in p['closure_roundoff']),
            inference='Original prediction-tail census classification only. Do not compute precision or recall '
                'of the single-background diagnostic from this selected subset or append old queries to the new sample.')
    result['elapsed_seconds']=time.perf_counter()-start
    output=REPO/'experiments/rl/results_single_background_reference_alignment_20261009.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output),seconds=result['elapsed_seconds'],tasks={task:dict(
        overlap=data['new_probability_sample_overlap'],tail=data['original_predicted_tail_census'],
        ledger_roundoff=data['largest_ledger_roundoff']) for task,data in result['tasks'].items()})))


if __name__=='__main__':
    main()
