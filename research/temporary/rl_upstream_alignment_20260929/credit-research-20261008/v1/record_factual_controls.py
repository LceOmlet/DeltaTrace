"""Bind completed passive controls and preserved PPO incident artifacts.

No model call, numerical tolerance, credit change, or training release.
"""
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from record_native_reference_evidence import binding, read, write, ROOT, RESULTS
from summarize_author_collection import quantiles

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]/'actor-nonfinite-20261008/v5'


def points(folder):
    rows = {}
    ranks = []
    for rank in (0, 1):
        report = read(folder/f'rank{rank}.json')
        assert report['phase'] == 'complete'
        ranks.append(report)
        for batch in report['batches']:
            for point in batch['points']:
                key = point['traj_uid'], point['packed_slot']
                assert key not in rows
                rows[key] = point
    assert len(rows) == 165
    return rows, ranks


def describe(rows):
    if not rows:
        return dict(points=0, states=0, trajectories=0)
    return dict(points=len(rows), states=len({p['initial_state_sha256'] for p in rows}),
        trajectories=len({p['traj_uid'] for p in rows}),
        abs_weighted_factual_drift_at_norm=quantiles([abs(p['boundaries']['32'][
            'DT_factual_minus_native_factual_effect']) for p in rows]),
        abs_matched_head_residual=quantiles([abs(p['matched_factual_residuals'][32]) for p in rows]),
        abs_factual_score_drift=quantiles([abs(p['DT_minus_native_factual_score']) for p in rows]),
        matched_decoder_sum=quantiles([math.fsum(p['matched_factual_residuals'][:32]) for p in rows]))


def main():
    stable_path = RESULTS/'results_stable_negative_credit_20261008.json'
    stable = read(stable_path)
    tasks = {}
    observations = []
    for task in ('textcraft', 'appworld'):
        folder = HERE/f'layer-factual-controls-{task}-observations'
        new, ranks = points(folder)
        old, _ = points(HERE/f'layer-{task}-observations')
        assert set(new) == set(old)
        history = [p for p in folder.glob('[0-9]*.json') if not read(p).get('files',{}).get(
            'results/rank0.json',{}).get('value',{}).get('summary_only')]
        observed_path = max(history, key=lambda p:read(p)['unix'])
        observed = read(observed_path)
        assert 'results/completed.json' in observed['files']
        assert not observed['files']['results/rank0.json']['value'].get('summary_only')
        observations.append(observed)
        launch_path = HERE/f'layer-factual-controls-{task}/launch.json'
        launch = read(launch_path)
        subsets = {}
        for cohort, value in stable['tasks'][task]['cohorts'].items():
            flags = {(p['traj_uid'],p['packed_slot']):p for p in value['points_with_identity']}
            subsets[cohort] = {flag:describe([new[k] for k,v in flags.items() if v[flag]])
                for flag in ('spurious_DT_tail_all_references', 'missed_native_tail_all_references',
                             'probability_bound_violated_all_references')}
        tasks[task] = dict(status='completed_passive_diagnostic_not_deployed',
            launch=binding(launch_path), pid=launch['pid'], birth=launch['birth'],
            devices=launch['devices'], base_commit_at_launch=launch['base_commit'],
            observer_sha256_at_launch=launch['scripts']['inspect_layer_collection.py'],
            source_path=launch['source_path'], source_sha256=launch['source_sha256'],
            ranks=[dict(receipt=binding(folder/f'rank{r}.json'),
                        actual_imports=record['owners'], operations=record['operations'],
                        elapsed_seconds=record['elapsed_seconds']) for r,record in enumerate(ranks)],
            unique_points=len(new), unchanged_frozen_identities=True,
            repeated_DT_exact_equal_points=sum(new[k]['fresh_DT_d']==old[k]['fresh_DT_d'] for k in new),
            repeated_native_exact_equal_points=sum(new[k]['native_single_d']==old[k]['native_single_d'] for k in new),
            repeated_native_absolute_d_difference=quantiles([abs(
                new[k]['native_single_d']-old[k]['native_single_d']) for k in new]),
            embedding_factual_control_maxabs=max(abs(p['boundaries']['0'][
                'DT_factual_minus_native_factual_effect']) for p in new.values()),
            matched_telescoping_roundoff_maxabs=max(abs(p['matched_telescoping_roundoff']) for p in new.values()),
            analysis=binding(HERE/f'layer-factual-controls-{task}-analysis.json'),
            robust_error_subsets=subsets, observation=binding(observed_path))
    terminal_folder=HERE/'layer-factual-controls-appworld-observations'
    terminal_path=max(terminal_folder.glob('[0-9]*.json'),key=lambda p:read(p)['unix'])
    terminal=read(terminal_path)
    value=dict(scope=__doc__, upstream_VERL_commit='20bd331',
        local_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_manifest=binding(HERE/'manifest.json'), stable_classifications=binding(stable_path),
        tasks=tasks, recorder=binding(Path(__file__)),
        conclusions=[
            'Weighted controls match the factual endpoint in diagnostic contractions only; no token credit is corrected.',
            'Opposing decoder/head residuals remain after factual matching. A residual of a whole decoder is not an isolated FA/FLA operator defect.',
            'These conditional descriptive distributions do not pool tail/body moments or establish a population error rate.',
            'No accepted numerical repair, changed Q/V/A, kernel tolerance assertion or task-performance improvement follows from this diagnostic alone.'
        ], production_modified=False, accepted_candidate=False, formal_training_restarted=False,
        checkpoint_restore=False, terminal_observation=binding(terminal_path),
        terminal_driver_alive=terminal['driver_alive'],
        physical_GPU_source='Original mx-smi in terminal_observation; sampled, not peak capacity acceptance',
        host_available_bytes=terminal['host_memory']['available'],
        observed_unix=max(terminal['unix'],*(o['unix'] for o in observations)))
    output=RESULTS/'results_factual_controls_20261008.json'
    write(output,value)

    preserved_path=ACTOR/'preserved-debug-latest.json'
    preserved=read(preserved_path)
    assert preserved['local_complete'] and preserved['update_records_complete']
    assert preserved['initial_state_complete'] and preserved['actual_update_input_complete']
    assert all(f['local_verified'] for f in preserved['files'])
    analysis_path=ACTOR/'native-input-gradient-analysis.json'
    analysis=read(analysis_path)
    assert not analysis['CUDA_initialized']
    assert all(len(r['steps'])==4 and all(not s['nonfinite_raw_gradients'] for s in r['steps'])
               for r in analysis['ranks'])
    actor=dict(scope='Complete original first-iteration diagnostic artifacts; original NaN remains unrepaired',
        manifest=binding(preserved_path), local_snapshot=preserved['files'][0]['local_path'],
        files=len(preserved['files']), bytes=preserved['total_bytes'],
        lossless_archive=preserved.get('lossless_archive'), actual_update_input_complete=True,
        initial_state_complete=True, update_records_complete=True,
        completed_optimizer_steps_per_rank=4, all_observed_raw_gradients_finite=True,
        analysis=binding(analysis_path), driver_pid=preserved['driver_pid'],
        driver_birth=preserved['driver_birth'], driver_alive_at_snapshot=preserved['driver_alive'],
        original_NaN_repaired=False, checkpoint_restore=False, formal_training_started=False)
    actor_output=RESULTS/'results_preserved_actor_debug_20261008.json'
    write(actor_output,actor)
    runtime=read(RESULTS/'current_runtime.json')
    runtime.update(observed_unix=value['observed_unix'],
        observed_utc=datetime.fromtimestamp(value['observed_unix'],timezone.utc).isoformat(),
        latest_observation_unix=value['observed_unix'],
        latest_observation_utc=datetime.fromtimestamp(value['observed_unix'],timezone.utc).isoformat(),
        latest_readonly_observation=dict(textcraft='Formal training stopped',appworld='Formal training stopped',
            diagnostic='Both passive factual-control collections complete; no accepted repair',
            receipt=binding(output)),
        latest_preserved_actor_debug_20261008=actor,
        latest_factual_controls_20261008=dict(receipt=binding(output),
            observed_unix=value['observed_unix'], status='completed_not_deployed',
            official_PPO_unchanged=True, Q_V_A_unchanged=True, accepted_candidate=False))
    runtime['latest_runtime_status_20261008'].update(observed_unix=value['observed_unix'],
        source_receipt=binding(terminal_path), research_processes_remaining=0,
        native_reference_diagnostics='Both factual-control probes complete; original PPO v5 also ended; no repair deployed',
        physical_resources='Bound terminal mx-smi: all8 GPUs 858/65536 MiB,0%,no GPU processes')
    write(RESULTS/'current_runtime.json',runtime)
    print(dict(result=binding(output),actor=binding(actor_output),
               points={t:v['unique_points'] for t,v in tasks.items()},preserved_files=len(preserved['files'])))


if __name__=='__main__':
    main()
