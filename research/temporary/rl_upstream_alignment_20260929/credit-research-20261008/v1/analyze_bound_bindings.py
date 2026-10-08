"""Join existing probability-bound observations, rebatching and actor positions.

CPU receipt analysis only. A probability-bound observation is not a component
intervention, a replacement advantage, or a measured per-position gradient.
"""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def reference(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def main():
    started = time.perf_counter()
    manifest = json.loads((HERE/'manifest.json').read_bytes())
    bounds = json.loads((HERE/'credit-probability-bounds.json').read_bytes())
    coefficients = json.loads((HERE/'collection-coefficients.json').read_bytes())
    assert bounds['provenance']['manifest.json']==reference(HERE/'manifest.json')['sha256']
    actor = {(p['traj_uid'],p['packed_slot']):p for p in coefficients['records']}
    result = dict(scope=__doc__, tasks={}, inputs=[reference(HERE/name) for name in
        ('manifest.json','credit-probability-bounds.json','collection-coefficients.json')])
    for task,data in bounds['tasks'].items():
        development = {u for g in manifest['tasks'][task]['groups']
                       if g['split']=='development' for u in g['trajectory_uids']}
        fresh = {}
        for rank in (0,1):
            path = HERE/f'layer-{task}-observations/rank{rank}.json'
            result['inputs'].append(reference(path))
            observation = json.loads(path.read_bytes())
            for batch in observation['batches']:
                for point in batch['points']:
                    key = (point['traj_uid'],point['packed_slot'])
                    assert key not in fresh
                    slot = batch['uids'].index(point['traj_uid'])
                    score = batch['DT_detail']['per_sample'][slot]['factual_target_logp']
                    fresh[key] = (point,score)
        assert len(fresh)==165
        assert {u for u,_ in fresh} <= development
        selected = []
        for point in data['probability_bound_violation_points']:
            assert point['traj_uid'] in development
            key = (point['traj_uid'],point['packed_slot'])
            row = dict(point)
            if key in fresh:
                f,score = fresh[key]
                assert point['source_index']==f['source_index']
                assert point['token_id']==f['token_id']
                row['existing_native_rebatch'] = dict(
                    fresh_DT_d=f['fresh_DT_d'],factual_DT_logp=score,
                    implied_deleted_logp=score-f['fresh_DT_d'],
                    native_single_d=f['native_single_d'],
                    native_factual_logp=f['factual_target_logp'],
                    native_deleted_logp=f['deleted_target_logp'])
            if task=='textcraft' and key in actor:
                original = actor[key]
                assert point['d']==original['d']
                assert point['token_id']==original['token_id']
                row['original_actor'] = {name:original[name] for name in (
                    'actor_matches','cohort','predicted_band','native_band',
                    'actual_raw_A','native_raw_A','delta_raw_A',
                    'delta_coefficient_fixed_scale')}
            selected.append(row)
        joined = [p for p in selected if 'existing_native_rebatch' in p]
        summary = {}
        for band in data['summary']:
            rows = [p for p in selected if p['stratum']==band]
            rebatches = [p for p in rows if 'existing_native_rebatch' in p]
            summary[band] = dict(total_saved_bound_observations=len(rows),
                existing_native_rebatch_positions=len(rebatches),
                fresh_positive_implied_deleted_logp=sum(
                    p['existing_native_rebatch']['implied_deleted_logp']>0 for p in rebatches),
                with_original_actor_mapping=sum('original_actor' in p for p in rows))
        result['tasks'][task] = dict(
            saved_bound_observations=len(selected),existing_rebatch_matches=len(joined),
            missing_development_captures=data['missing_uids'],strata=summary,points=selected)
    result.update(source=reference(Path(__file__)), elapsed_seconds=time.perf_counter()-started,
        operations=dict(model=0,DT=0,GPU=0,gradient=0,update=0,production_modified=False),
        limitations=[
            'Old and fresh DT scores use their own captured factual logp; batching drift is retained, not corrected.',
            'A positive implied logp is a mathematical constraint violation, not an official numerical-tolerance failure. Magnitude must remain visible, especially near zero.',
            'Actor positions and fixed-moment coefficient errors are not per-position gradient measurements or an explanation of the historical degradation.',
            'The existing measured native subset is frozen. Unmatched observations are not assigned a native value, tested status or gradient.'])
    (HERE/'probability-bound-bindings.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({task:value['strata'] for task,value in result['tasks'].items()}))


if __name__=='__main__':
    main()
