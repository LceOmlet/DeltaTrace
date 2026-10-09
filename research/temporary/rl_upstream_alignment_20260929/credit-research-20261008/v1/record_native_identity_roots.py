"""Summarize completed native null roots without inventing a numeric tolerance."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    data=path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\','/'),
        bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    folder=HERE/'native-identity-roots-appworld-v1'
    launch=json.loads((folder/'launch.json').read_bytes())
    ranks=[json.loads((folder/f'rank{r}.json').read_bytes()) for r in (0,1)]
    completed=json.loads((folder/'completed.json').read_bytes())
    spec=json.loads((HERE/'layer-collection-inputs.json').read_bytes())['tasks']['appworld']
    entries={e['traj_uid']:e for e in spec['entries']}
    assert all(r['phase']=='complete' and len(r['batches'])==6 for r in ranks)
    assert all(r['completed'] and r['optimizer_steps']==0 for r in completed)
    batches=sorted((b for r in ranks for b in r['batches']),key=lambda b:b['index'])
    assert [b['index'] for b in batches]==list(range(12))
    points=[]
    for b in batches:
        assert len(b['rows'])==4 and len(b['duplicated_cache_fields'])==64
        assert all(all(f['pair_equal']) for f in b['duplicated_cache_fields'])
        for row,r in enumerate(b['rows']):
            assert r['input_pair_equal'] and all(r['masks_pair_equal'].values()) and r['positions_pair_equal']
            assert len(r['boundaries'])==33 and r['boundaries'][0]['equal']
            e=entries[r['traj_uid']]
            points.append(dict(r,batch=b['index'],batch_row=row,
                initial_state_sha256=e['initial_state_sha256'],
                first_unequal_boundary=b['first_unequal_boundary_by_row'][row],
                factual_minus_reference=r['factual_logp']-r['reference_logp']))
    assert len(points)==48 and len({p['traj_uid'] for p in points})==48
    unequal=[p for p in points if p['first_unequal_boundary'] is not None]
    score_only=[p for p in points if p['first_unequal_boundary'] is None and p['factual_minus_reference']!=0]
    oldpath=HERE/'saved-identity-controls-v1/result.json'
    old=json.loads(oldpath.read_bytes())['tasks']['appworld']['rows']
    by_uid={p['traj_uid']:p for p in points}
    comparisons=[]
    for r in old:
        if not r['active_query'] and r['root_effect']!=0:
            p=by_uid[r['traj_uid']]
            comparisons.append(dict(traj_uid=r['traj_uid'],old_batch=r['batch'],old_round=r['round'],
                old_native_effect=r['root_effect'],all_identity_native_effect=p['factual_minus_reference'],
                difference=p['factual_minus_reference']-r['root_effect']))
    assert len(comparisons)==19
    observations=[]
    for path in sorted(folder.glob('observation-*.json')):
        x=json.loads(path.read_bytes())
        observations.append((path,x))
    terminal=[(p,x) for p,x in observations if not x.get('driver_alive',True) and 'results/completed.json' in x.get('files',{})]
    assert terminal
    lastpath,last=terminal[-1]
    worker_pids={r['pid'] for r in ranks}
    pss=[p['pss'] for _,x in observations for p in x.get('processes',[])
         if p['pid'] in worker_pids and p.get('pss') is not None]
    result=dict(
        role='Completed passive native-root diagnostic, not a method, attribution repair or official tolerance pass',
        source_commit=launch['code_commit'],script=ref(Path(__file__)),
        inputs=[ref(folder/'launch.json'),ref(folder/'rank0.json'),ref(folder/'rank1.json'),
                ref(folder/'completed.json'),ref(lastpath),ref(oldpath),ref(HERE/'layer-collection-inputs.json')],
        frame=dict(task='appworld',original_B4=12,distinct_trajectories=48,
            initial_states=len({p['initial_state_sha256'] for p in points}),
            scope='Entire already-frozen layer-collection AppWorld frame, not a new task-population sample or the 1061-point recall frame'),
        paired_input_integrity=dict(all_ID_mask_position_pairs_equal=True,
            original_cache_fields_per_B4=64,all_duplicated_cache_field_pairs_equal=True,
            all_embedding_boundary_pairs_equal=True),
        hidden_unequal=dict(trajectories=len(unequal),initial_states=len({p['initial_state_sha256'] for p in unequal}),
            first_boundary_counts={str(k):sum(p['first_unequal_boundary']==k for p in unequal) for k in sorted({p['first_unequal_boundary'] for p in unequal})},
            maximum_absolute_joint_logp_difference=max(abs(p['factual_minus_reference']) for p in unequal)),
        score_nonzero_without_recorded_hidden_difference=[{k:p[k] for k in ['traj_uid','batch','factual_minus_reference']} for p in score_only],
        old_inactive_control_reproduction=comparisons,
        maximum_absolute_old_control_reproduction_difference=max(abs(p['difference']) for p in comparisons),
        operations=[r['operations'] for r in ranks],
        resource=dict(worker_elapsed_seconds=[r['seconds'] for r in ranks],
            wall_launch_to_latest_worker_complete_seconds=max(r['unix'] for r in ranks)-launch['launched_unix'],
            maximum_sampled_worker_PSS_bytes=max(pss) if pss else None,
            scope='PSS is a sampled lower bound on the peak. Physical VRAM snapshots are saved, not a measured peak. Original prefix preparation is included in worker elapsed time.',
            terminal_driver_alive=last['driver_alive'],terminal_physical_receipt=ref(lastpath)),
        actual_numerical_owners=[r['actual_numerical_owners'] for r in ranks],
        points=points,
        conclusions=[
            'The nonzero inactive control endpoint effects recur with all factual/reference IDs identical. They are not an effect of deleting the queried token or an extra downstream credit allocation.',
            'The first unequal recorded hidden boundary follows GDN decoder 0, 1 or 2. This brackets a native decoder computation; it does not identify FLA, convolution, RMS, projection or residual addition individually.',
            'Actual duplicated IDs, masks, positions, cache fields and initial embeddings are equal. Later joint-score differences can therefore include a native numeric/background component.',
            'No original FA/FLA actual-dtype comparison was performed on these first-divergent operands. This is not evidence of an official tolerance violation and does not authorize a compensation multiplier.',
            'Large negative credit supported by deletion counterfactuals remains unchanged. This investigation concerns estimation/reference discrepancy, not the existence of negative tails.'
        ],
        limitations=[
            'The observer saved per-boundary statistics, exact source/input artifact bindings and commands, but did not persist the first-divergent operator Q/K/V tensors. Operator tolerance requires compatible existing captures or one bounded operand capture; do not claim a pass from these statistics.',
            'The final norm output and logsoftmax internals were not separately observed. Tiny score differences with equal recorded hidden states are kept separate.',
            'This diagnostic does not prove a cause of all DT approximation error, policy degradation or original PPO NaN.'
        ],production_modified=False,official_tolerance_changed=False,training_restart=False,
        extreme_credit_accuracy_repaired=False,original_PPO_NaN_repaired=False)
    output=REPO/'experiments/rl/results_native_identity_roots_20261009.json'
    output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({k:result[k] for k in ['frame','hidden_unequal','maximum_absolute_old_control_reproduction_difference','resource']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
