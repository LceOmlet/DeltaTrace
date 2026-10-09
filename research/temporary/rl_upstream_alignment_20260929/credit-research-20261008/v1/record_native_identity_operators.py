"""Record operator-local evidence without broadening numeric test claims."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    data=path.read_bytes()
    return dict(path=str(path.relative_to(REPO)).replace('\\','/'),bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def main():
    folder=HERE/'native-identity-operators-appworld-v2'
    failed=HERE/'native-identity-operators-appworld-v1'
    launch=json.loads((folder/'launch.json').read_bytes())
    ranks=[json.loads((folder/f'rank{i}.json').read_bytes()) for i in (0,1)]
    complete=json.loads((folder/'completed.json').read_bytes())
    check=json.loads((folder/'linear-roundoff.json').read_bytes())
    check_launch=json.loads((folder/'linear-check-launch.json').read_bytes())
    assert all(r['phase']=='complete' and r['operations']['native_root']==2 for r in ranks)
    assert all(r['completed'] and r['optimizer_steps']==0 for r in complete)
    assert check['status']=='complete' and not check['cuda_initialized']
    observations=[(p,json.loads(p.read_bytes())) for p in sorted(folder.glob('observation-*.json'))]
    terminal_path,terminal=observations[-1]
    assert not terminal['driver_alive'] and 'results/linear-roundoff.json' in terminal['files']
    oldpath=REPO/'experiments/rl/results_native_identity_roots_20261009.json'
    old={p['traj_uid']:p for p in json.loads(oldpath.read_bytes())['points']}
    batches=sorted((b for r in ranks for b in r['batches']),key=lambda b:b['index'])
    assert [b['index'] for b in batches]==[0,1,10,11]
    spec=json.loads((HERE/'layer-collection-inputs.json').read_bytes())['tasks']['appworld']
    entries={e['traj_uid']:e for e in spec['entries']}
    points=[];operation_refs=[]
    for batch in batches:
        rank=0 if batch['index'] in (0,10) else 1
        path=folder/f"rank{rank}-batch{batch['index']}-operators.json"
        operation_refs.append(ref(path));operators=json.loads(path.read_bytes())
        for row,r in enumerate(batch['rows']):
            previous=old[r['traj_uid']]
            score=r['factual_logp']-r['reference_logp']
            first=operators['first_unequal_by_row'].get(str(row))
            event=operators['events'][first['event']] if first is not None else None
            if event is not None:
                assert all(s['equal'] for s in event['inputs']['input']['rows'])
            points.append(dict(traj_uid=r['traj_uid'],batch=batch['index'],row=row,
                initial_state_sha256=entries[r['traj_uid']]['initial_state_sha256'],
                root_score_difference=score,previous_root_score_difference=previous['factual_minus_reference'],
                repeated_root_difference=score-previous['factual_minus_reference'],
                first_unequal=first,first_unequal_output=event['output']['rows'][row] if event is not None else None,
                first_operation_input_all_pairs_equal=True if event is not None else None))
    pair_checks=[s for c in check['cases'] for s in c['native_pair_checks']]
    scalar_checks=[s for c in check['cases'] for r in c['all_unequal_cells_vs_FP64_rounded_BF16_checks'] for s in r['endpoints']]
    assert len(pair_checks)==12 and len(scalar_checks)==22
    assert all(s['status']=='passed' for s in pair_checks+scalar_checks)
    assert len({tuple(s['rtol_atol']) for s in pair_checks+scalar_checks})==1
    unequal=[p for p in points if p['first_unequal'] is not None]
    result=dict(role='Completed native operator localization and original PyTorch dtype comparison; not a production repair',
        inputs=[ref(folder/p) for p in ('launch.json','rank0.json','rank1.json','completed.json','linear-roundoff.json','linear-check-launch.json')]
            +operation_refs+[ref(terminal_path),ref(oldpath)],
        failed_observer_attempt=dict(launch=ref(failed/'launch.json'),rank1=ref(failed/'rank1.json'),
            terminal=ref(failed/'observation-1791552121.json'),source_commit='1f2b872f',accepted=False,
            issue='Diagnostic state_dict export was followed by a mixed Tensor/DTensor error in the next original Linear call. It is not a production failure. The exact FSDP side-effect hook was not separately traced.',
            correction='Observer v2 reads the original Linear return frame through LocalCaptureEvents; no state_dict call. The original endpoint score differences reproduce.',
            numerical_results_after_failure_accepted=False),
        source_commit=launch['code_commit'],check_source_commit=check_launch['code_commit'],
        numerical_version=launch['protocol']['numerical_version'],
        actual_numerical_owners=[r['actual_numerical_owners'] for r in ranks],
        frame=dict(original_B4_indices=[0,1,10,11],trajectories=len(points),
            initial_states=len({p['initial_state_sha256'] for p in points}),
            scope='Operator localization of all eight previously unequal native-root rows plus original matched control batches. Not a quality/development sample or replacement recall frame.'),
        first_unequal=dict(trajectories=len(unequal),operations={name:sum(p['first_unequal']['operation']==name for p in unequal)
            for name in sorted({p['first_unequal']['operation'] for p in unequal})}),
        original_endpoint_reproduction_maxabs=max(abs(p['repeated_root_difference']) for p in points),
        torch_checks=dict(owner='Original installed torch.nn.Linear/F.linear and torch.testing.assert_close, Torch '+check['torch_version'],
            owner_sources=check['owners'],rtol_atol=pair_checks[0]['rtol_atol'],
            full_native_pair_checks=len(pair_checks),all_unequal_cell_endpoint_checks=len(scalar_checks),
            total_unequal_cells=sum(len(c['unequal_elements']) for c in check['cases']),status='passed',
            scope='BF16 defaults from the owning public PyTorch assertion; not FA/FLA tolerances or an original whole-model/VERL test. FP64 reference is evaluated only at every unequal coordinate, not at the equal coordinates.'),
        resource=dict(worker_elapsed_seconds=[r['seconds'] for r in ranks],
            capture_launch_to_last_worker_complete_seconds=max(r['unix'] for r in ranks)-launch['launched_unix'],
            worker_terminal_PSS_bytes=[r['process_pss_bytes'] for r in ranks],
            CPU_check_seconds=check['seconds'],CPU_check_terminal_PSS_bytes=check['sampled_terminal_PSS_bytes'],
            CPU_check_cuda_initialized=check['cuda_initialized'],
            terminal_host_available_bytes=terminal['host']['available'],terminal_disk_free_bytes=terminal['disk']['free'],
            physical_receipt=ref(terminal_path),scope='Terminal and sampled readings, not true peaks. CPU check performs no model or GPU work.'),
        handles=dict(driver=dict(pid=launch['pid'],birth=launch['birth'],alive=False),
            workers=[dict(pid=r['pid'],birth=r['birth'],phase=r['phase']) for r in ranks],
            CPU_check=dict(pid=check['pid'],birth=check['birth'],status=check['status'])),
        operations=[r['operations'] for r in ranks],points=points,
        conclusions=[
            'All eight unequal root rows first diverge in original native Linear projections, before the corresponding convolution and FLA. Later FLA operand/output differences cannot by themselves prove a FLA defect.',
            'The captured Linear differences pass the owning PyTorch assertion with unchanged BF16 defaults. No projection, DT, FA/FLA or credit correction is deployed.',
            'A permitted local floating-point discrepancy can propagate to a nonzero joint-score difference. Local checks neither bound that whole-model difference nor establish an attribution error for every queried source.',
            'Original deletion-supported negative tails remain unchanged. The same-input endpoint discrepancy is separate from an actual deletion effect.',
            'The prior 48-row investigation does not need to be repeated to obtain these operands: complete original artifacts are now saved remotely and bound by SHA256.'
        ],production_modified=False,official_tolerance_changed=False,training_restart=False,
        attribution_accuracy_repaired=False,original_PPO_NaN_repaired=False)
    output=REPO/'experiments/rl/results_native_identity_operators_20261009.json'
    output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({k:result[k] for k in ('frame','first_unequal','original_endpoint_reproduction_maxabs','torch_checks','resource')},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
