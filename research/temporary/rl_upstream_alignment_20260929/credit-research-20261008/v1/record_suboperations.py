"""Bind passive suboperation results or running status; no repair acceptance."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tarfile

from record_native_reference_evidence import binding, read, write, RESULTS

HERE = Path(__file__).resolve().parent


def main():
    # Preserve the failed diagnostic's exact source, not the corrected current
    # working files. It was a report traversal defect, not a DT/PPO failure.
    failed = HERE/'layer-suboperations-textcraft'
    saved = failed/'failed-observer-source'
    saved.mkdir(exist_ok=True)
    with tarfile.open(failed/'observer-source.tar') as archive:
        for name in ('inspect_layer_collection.py','passive_suboperations.py'):
            (saved/name).write_bytes(archive.extractfile(name).read())
    failure_observations = sorted((HERE/'layer-suboperations-textcraft-observations').glob('[0-9]*.json'))
    failure = max(failure_observations,key=lambda p:read(p)['unix'])
    tasks = {}
    newest = []
    for task in ('textcraft','appworld'):
        tag = 'layer-suboperations-'+task+'-v2'
        launch_path = HERE/tag/'launch.json'
        launch = read(launch_path)
        folder = HERE/(tag+'-observations')
        observed_path = max(folder.glob('[0-9]*.json'),key=lambda p:read(p)['unix'])
        observed = read(observed_path)
        newest.append(observed)
        analysis_path = HERE/(tag+'-analysis.json')
        complete = analysis_path.exists() and read(analysis_path)['complete']
        ranks = [read(folder/f'rank{r}.json') for r in (0,1)] if complete else []
        analysis = read(analysis_path) if complete else None
        tasks[task] = dict(status='completed_passive_diagnostic' if complete else 'running_passive_diagnostic',
            launch=binding(launch_path), observation=binding(observed_path),
            PID=launch['pid'], birth=launch['birth'], devices=launch['devices'],
            source_path=launch['source_path'], source_sha256=launch['source_sha256'],
            base_commit_at_launch=launch['base_commit'], source_patch_hashes=launch['scripts'],
            actual_owner_imports=[r['owners'] for r in ranks] if complete else 'Pending completed collection receipt',
            analysis=binding(analysis_path) if complete else None,
            completed_unique_points=analysis['unique_points'] if complete else None,
            operations=[r['operations'] for r in ranks] if complete else None,
            measured_elapsed_seconds=analysis['elapsed_seconds'] if complete else None,
            repeated_DT_exact_equal_points=sum(c['DT_difference']==0 for c in analysis['unchanged_output_comparisons']) if complete else None,
            repeated_native_max_abs_difference=max(abs(c['native_difference']) for c in analysis['unchanged_output_comparisons']) if complete else None,
            maximum_retained_host_bank_bytes=analysis['maximum_observed_retained_host_bank_bytes'] if complete else None,
            additional_readout_seconds=analysis['diagnostic_readout_seconds'] if complete else None,
            credit_repaired=False, candidate=False, formal_release=False)
    value = dict(scope=__doc__, upstream_VERL_commit='20bd331',
        source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_inputs=binding(HERE/'layer-collection-inputs.json'), protocol=binding(HERE/'suboperation-protocol.json'),
        stable_classifications=binding(RESULTS/'results_stable_negative_credit_20261008.json'),
        PPO_debug_preservation=binding(RESULTS/'results_preserved_actor_debug_20261008.json'),
        original_PPO_NaN_repaired=False,
        failed_observer_v1=dict(launch=binding(failed/'launch.json'), observation=binding(failure),
            sources=[binding(saved/name) for name in ('inspect_layer_collection.py','passive_suboperations.py')],
            observed_operations_per_rank=[read(failure)['files'][f'results/rank{r}.json']['value']['operations'] for r in (0,1)],
            defect="Report loop included its own previous mode's formatted dictionary, producing KeyError('matched').",
            remedy='Filter contraction records by actual mode; persist raw point before optional formatting.',
            classification='diagnostic formatter defect; not model/DT numerical failure',
            production_modified=False),
        report_only_CPU_check=binding(HERE/'report-formatter-cpu-check.json'), tasks=tasks,
        source_files={name:binding(HERE/name) for name in ('passive_suboperations.py','inspect_layer_collection.py',
            'prepare_suboperation_protocol.py','analyze_suboperations.py','record_suboperations.py')},
        conclusions=[
            'Passive suboperation callbacks return the original owner objects; production Q/V/A, EOS endpoints, source selection, PPO and whitening are unchanged.',
            'Residual additions and actual-versus-recomputed output coefficients remain separate. Neither BF16 residual addition nor head seed reconstruction is assumed exact.',
            'Large opposing terms must not be subtracted as a repair. This is localization of approximation, not an isolated FA/FLA kernel failure or an accepted method candidate.',
            'Task/cohort, crossed prediction/reference ratio cells and exposure remain separate. No pooled tail/body moments or population error claim.',
            'No new PPO replay, checkpoint restoration, official-tolerance claim or formal restart.'
        ], production_modified=False, numerical_rules_changed=False, official_tolerance_changed=False,
        accepted_candidate=False, formal_training_restarted=False, checkpoint_restore=False,
        observed_unix=max(o['unix'] for o in newest), recorder=binding(Path(__file__)))
    output = RESULTS/'results_suboperations_20261008.json'
    write(output,value)
    runtime = read(RESULTS/'current_runtime.json')
    when = datetime.fromtimestamp(value['observed_unix'],timezone.utc).isoformat()
    runtime.update(observed_unix=value['observed_unix'], observed_utc=when,
        latest_observation_unix=value['observed_unix'], latest_observation_utc=when,
        latest_suboperations_20261008=dict(receipt=binding(output), tasks={t:x['status'] for t,x in tasks.items()},
            accepted_candidate=False, official_PPO_unchanged=True,Q_V_A_unchanged=True),
        latest_readonly_observation=dict(textcraft='Formal training stopped',appworld='Formal training stopped',
            diagnostic='Passive suboperation diagnostic; see per-task completed/running status',receipt=binding(output)))
    runtime['latest_runtime_status_20261008'].update(observed_unix=value['observed_unix'],
        research_processes_remaining=sum(x['status']=='running_passive_diagnostic' for x in tasks.values()),
        native_reference_diagnostics='Suboperation diagnostics only; no repair deployed or formal restart',
        physical_resources='Use this observation phase/resource receipt; previous terminal idle snapshot is historical, not current occupancy.',
        source_receipt=binding(output))
    write(RESULTS/'current_runtime.json',runtime)
    print(dict(receipt=binding(output), tasks={t:x['status'] for t,x in tasks.items()}))


if __name__=='__main__':
    main()
