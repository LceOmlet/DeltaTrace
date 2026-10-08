"""Record completed frozen diagnostics, preserving failed runs as failed.

This records evidence only. It does not change the estimator, deployment,
training entry, original numerical tolerances, or the frozen query population.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    analysis_path = HERE/'single-background-appworld-range-merged-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    text_path = HERE/'single-background-textcraft-analysis.json'
    text = json.loads(text_path.read_bytes())
    assert analysis['complete'] and analysis['point_count'] == analysis['expected_points'] == 165
    assert not analysis['missing_points']
    assert text['complete'] and text['point_count'] == 165
    for artifact in analysis['inputs']:
        assert ref(Path(artifact['path']))['sha256'] == artifact['sha256']
    folder = HERE/'single-background-range-resume-observations'
    observation_path = folder/'1791484124.json'
    observation = json.loads(observation_path.read_bytes())
    assert not observation['driver_alive']
    ranks = [json.loads((folder/f'rank{rank}.json').read_bytes()) for rank in (0, 1)]
    assert all(r['phase'] == 'complete' and r['operations']['DT'] == 20 for r in ranks)
    assert all(all(v == 0 for k, v in r['operations'].items() if k != 'DT') for r in ranks)
    launch_path = HERE/'single-background-range-resume-launch.json'
    launch = json.loads(launch_path.read_bytes())
    manifest_path = HERE/'range-owner-resume/preserved-artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    archive = Path(manifest['local_archive'])
    assert ref(archive)['sha256'] == manifest['archive']['sha256']
    # Check the exact archived bytes without extracting long Windows paths.
    with tarfile.open(archive, 'r:gz') as saved:
        members = {member.name: member for member in saved.getmembers() if member.isfile()}
        assert set(members) == {item['name'] for item in manifest['files']}
        for item in manifest['files']:
            payload = saved.extractfile(members[item['name']]).read()
            assert len(payload) == item['bytes']
            assert hashlib.sha256(payload).hexdigest() == item['sha256']
    tasks = {}
    for name, report in [('textcraft', text), ('appworld', analysis)]:
        points = report['points']
        false = [p for p in points if p['original_spurious_tail']]
        missed = [p for p in points if p['original_missed_tail'] and 'uniform' in p['cohorts']]
        tasks[name] = dict(
            complete=True, frozen_unique_positions=len(points),
            originally_spurious_negative_tail=len(false),
            initial_states_with_spurious_tail=len({p['initial_state_sha256'] for p in false}),
            still_spurious_negative_tail=sum(p['single_still_spurious_tail'] for p in false),
            residual_wrong_sign_among_original_spurious=sum(p['single_opposite_sign'] for p in false),
            uniform_originally_missed_native_tail=len(missed),
            uniform_missed_tail_still_outside_predicted_tail=sum(p['single_bin'] in ['ratio_le_1', 'ratio_1_to_2'] for p in missed),
            uniform_missed_tail_comparisons=[dict(traj_uid=p['traj_uid'], packed_slot=p['packed_slot'],
                single_d=p['single_d'], single_bin=p['single_bin'], native_bin=p['native_bin'],
                native_interval=p['native_interval']) for p in missed],
            cross_cell_analysis=ref(text_path if name == 'textcraft' else analysis_path),
            aggregation=report['weighting'])
    output = REPO/'experiments/rl/results_single_background_completed_20261009.json'
    result = dict(
        status='frozen_diagnostic_collection_complete_not_training_repair',
        scope='Single-background reference diagnosis for the frozen TextCraft/AppWorld collection; no production single-source refinement.',
        tasks=tasks,
        original_partial_appworld_receipt=ref(REPO/'experiments/rl/results_single_background_appworld_20261009.json'),
        numerical_boundary_replay_receipt=ref(REPO/'experiments/rl/results_fla_range_owner_20261009.json'),
        merged_appworld_parts=dict(original_partial_positions=74, failed_call_replay_positions=8,
                                  missing_query_continuation_positions=83, total_unique_positions=165,
                                  original_part_phases=analysis['part_phases']),
        continuation=dict(launch=ref(launch_path), pid=launch['pid'], birth=launch['birth'],
            observation=ref(observation_path), phase='complete',
            original_code_commit_at_launch=launch['code_commit'],
            exact_script_sha256=launch['scripts']['inspect_single_background_collection.py'],
            recorded_code_commit='3852dbb6',
            original_source_sha256=launch['source_sha256'],
            runner=launch['actual_recorded_runner'], owner=launch['isolated_range_owner'],
            ranks=[dict(rank=r['rank'], pid=r['pid'], birth=r['birth'],
                elapsed_seconds=r['elapsed_seconds'], operations=r['operations'],
                phase=r['phase'], points=sum(len(batch['points']) for batch in r['batches'])) for r in ranks]),
        preservation=dict(manifest=ref(manifest_path), local_archive=ref(archive),
            files=len(manifest['files']), bytes=sum(item['bytes'] for item in manifest['files']),
            every_archive_member_SHA256_verified=True),
        mechanistic_result='Changing only the diagnostic reference background removes the original spurious negative-tail classification in 18 TextCraft and 11 AppWorld cases, across six initial states per task. This identifies a joint-background allocation contribution; finite propagation and native endpoint discrepancies remain separate.',
        limits='No population-tail moment claim, no isolated GDN dominance claim, no replacement of author cumulative deletion/RISE/MAS, no proof that single-background estimates are uniformly accurate, no training-effect conclusion.',
        resources=dict(observation=ref(observation_path), host_available_bytes=observation['host_memory']['available'],
            scope='Terminal physical observation, not peak training or DT memory; diagnostic driver exited.'),
        formal_restart=False, checkpoint_restore=False, production_modified=False,
        QVA_modified=False, PPO_modified=False, attribution_accuracy_repair_accepted=False,
        original_PPO_NaN_fixed=False,
        recorder=ref(Path(__file__)),
        source_api_mapping=ref(HERE/'gdn-context-algebra-cost.json'))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    path = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(path.read_bytes())
    current['observed_unix'] = observation['unix']
    current['observed_utc'] = datetime.fromtimestamp(observation['unix'], timezone.utc).isoformat()
    current['latest_single_background_completed_20261009'] = dict(
        receipt=ref(output), phase='complete', tasks=tasks, formal_restart=False,
        production_modified=False, attribution_accuracy_repair_accepted=False)
    current['latest_fla_range_owner_20261009']['continuation_completed'] = dict(
        receipt=ref(output), observed_unix=observation['unix'], unique_positions=165)
    current['latest_readonly_observation'] = dict(
        textcraft='Formal training stopped', appworld='Formal training stopped',
        diagnostic='Frozen single-background comparison complete for both tasks; joint-background allocation is implicated. Internal accuracy repair remains unaccepted.',
        receipt=ref(output))
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(receipt=str(output), tasks=tasks, archive_verified=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
