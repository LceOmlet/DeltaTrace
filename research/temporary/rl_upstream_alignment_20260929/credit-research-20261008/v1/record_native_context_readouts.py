"""Bind original-operator API/cost evidence without accepting a credit repair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    result_path = HERE/'native-context-readouts-v5-result.json'
    results = json.loads(result_path.read_bytes())
    profile_path = HERE/'native-context-profile-result.json'
    profile = json.loads(profile_path.read_bytes())
    invalid = {'raw shifted-query first o', 'raw shifted-query after replay o'}
    checks, controls = [], []
    for case in results['cases']:
        assert case['initial_state_nonzero'] and case['reverse_boundary_nonzero']
        assert case['external_layouts']['k']['contiguous'] is False
        assert case['actual_layouts']['k']['stride'] == [131072, 1024, 128, 1]
        for check in case['checks']:
            row = dict(dtype=case['dtype'], **check)
            (controls if check['name'] in invalid else checks).append(row)
    assert len(checks) == 12 and all(c['status'] == 'passed' for c in checks)
    assert len(controls) == 4 and all(c['status'] == 'failed' for c in controls)
    preserved = {}
    for version in ('v1', 'v3', 'v5', 'profile'):
        path = HERE/f'native-context-{version}/preserved-artifacts/manifest.json'
        manifest = json.loads(path.read_bytes())
        assert manifest['all_file_SHA256_verified']
        assert ref(Path(manifest['local_archive']))['sha256'] == manifest['archive']['sha256']
        preserved[version] = dict(manifest=ref(path), files=len(manifest['files']),
            bytes=sum(item['bytes'] for item in manifest['files']),
            archive=manifest['archive'], local_archive=manifest['local_archive'],
            all_file_SHA256_verified=True)
    observation_folder = HERE/'native-context-profile-observations'
    observation_path = max(observation_folder.glob('[0-9]*.json'), key=lambda p:int(p.stem))
    observation = json.loads(observation_path.read_bytes())
    assert not observation['driver_alive']
    output = REPO/'experiments/rl/results_native_context_readouts_20261009.json'
    payload = dict(
        status='original_state_readout_composition_and_primitive_cost_verified_not_credit_repair',
        observable_need='Reuse native factual past/future states for conditional DT propagation without per-source model forwards or a dense per-time state bank.',
        method_scope=ref(REPO/'experiments/rl/PLAN.md'),
        motivating_collection=ref(REPO/'experiments/rl/results_single_background_completed_20261009.json'),
        original_readout=results['original_readout'], original_test=results['original_test'],
        actual_dtype_checks=dict(scope=results['limits'], result=ref(result_path),
            correct_compositions_checks_passed=12, invalid_layout_controls_failed=4,
            original_thresholds_changed=False, checks=checks, controls=controls,
            reference='Original FLA 0.4.1 recurrent FP32 reference on exact low-precision operands and represented seed; original assert_close for o/dq/dv only.'),
        diagnosed_interface_error=dict(
            scope='New isolated diagnostic only; original public FLA path already uses input_guard.',
            raw_key_stride=[131072,1,16384,128], native_key_stride=[131072,1024,128,1],
            symptom='Direct low-level readout on the raw non-contiguous GPU key gives normalized output error approximately 0.516; reuse captured native contiguous key gives original o tolerance.',
            preserved_failed_v1=ref(HERE/'native-context-readouts-v1-result.json'),
            layout_ablation_result=ref(result_path),
            owner_source=ref(HERE/'native-context-owner-source.json'),
            low_level_source=ref(HERE/'native-chunk-readout-source.json'),
            existing_training_defect=False,
            conclusion='Compose the native owner with its required layouts. Do not infer contiguity from logical shape, equal values, or a prior CPU contiguous call.'),
        diagnostic_startup_failure=dict(
            version='v3', scope='No numerical test or model/DT call occurred.',
            cause='Version-string replacement accidentally changed the recorded operands path from the nonfinite v2 directory to v3; failed at CPU torch.load. Original command/log preserved; later launcher verifies named input files before creating a process.',
            launch=ref(HERE/'native-context-readouts-v3-launch.json')),
        primitive_profile=dict(result=ref(profile_path), original_owners=dict(
            finite_owner=profile['finite_owner'], native_readout=profile['native_readout']),
            factual_shape=profile['factual_shape'], dtype=profile['dtype'],
            timings=profile['timings'],
            warm_median_seconds={name:statistics.median(item['repeated_seconds'])
                                 for name,item in profile['timings'].items()},
            preparation_seconds=profile['preparation_seconds'],
            peak_allocated_bytes=profile['peak_allocated_bytes'], peak_reserved_bytes=profile['peak_reserved_bytes'],
            limits=profile['limits']),
        preservation=preserved,
        versions=dict(v1_launch=ref(HERE/'native-context-readouts-launch.json'),
            v2_launch=ref(HERE/'native-context-readouts-v2-launch.json'),
            v3_launch=ref(HERE/'native-context-readouts-v3-launch.json'),
            v4_launch=ref(HERE/'native-context-readouts-v4-launch.json'),
            v5_launch=ref(HERE/'native-context-readouts-v5-launch.json'),
            profile_launch=ref(HERE/'native-context-profile-launch.json')),
        remaining_work='Complete both conditional finite endpoint orientations and the coupled width4 conv coefficients inside the DT owner, enumerate actual query count and live tensors, then use frozen collection and author cumulative deletion/RISE/MAS. This API test is not a candidate estimator or an isolated GDN dominance claim.',
        source_mapping=ref(HERE/'gdn-context-algebra-cost.json'),
        observation=ref(observation_path), production_modified=False, formal_restart=False,
        checkpoint_restore=False, QVA_modified=False, PPO_modified=False,
        attribution_accuracy_repair_accepted=False, model_calls=0, DT_calls=0, optimizer=0,
        recorder=ref(Path(__file__)))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    path = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(path.read_bytes())
    current['observed_unix'] = observation['unix']
    current['observed_utc'] = datetime.fromtimestamp(observation['unix'], timezone.utc).isoformat()
    current['latest_native_context_readouts_20261009'] = dict(receipt=ref(output),
        status=payload['status'], production_modified=False, formal_restart=False,
        attribution_accuracy_repair_accepted=False)
    current['latest_readonly_observation'] = dict(textcraft='Formal training stopped',
        appworld='Formal training stopped', diagnostic=payload['status'], receipt=ref(output))
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(receipt=str(output), valid_checks=12, invalid_layout_controls=4,
        warm_medians=payload['primitive_profile']['warm_median_seconds'], accepted_credit_repair=False)))


if __name__ == '__main__':
    main()
