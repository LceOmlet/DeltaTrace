"""Record the isolated owner composition without accepting a whole DT repair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    query_path = HERE/'native-conditional-queries-result.json'
    limit_path = HERE/'conditional-memory-limit-result.json'
    finite_path = HERE/'conditional-memory-finite-result.json'
    query, limit, finite = [json.loads(path.read_bytes()) for path in (query_path, limit_path, finite_path)]
    assert len(query['cases']) == len(limit['cases']) == len(finite['cases']) == 2
    for outcome in (query, limit):
        assert all(row['status'] == 'passed' for case in outcome['cases'] for row in case['checks'])
    assert sum(len(case['checks']) for case in query['cases']) == 10
    assert sum(len(case['checks']) for case in limit['cases']) == 28
    assert all(case['original_readout_calls'] == 40 for case in limit['cases']+finite['cases'])
    assert all(row['status'] == 'passed' for case in finite['cases'] for row in case['original_output_checks'])
    assert sum(len(case['original_output_checks']) for case in finite['cases']) == 26
    assert all(case['sources_per_trajectory'] == 199 and case['finite_scalar']['nonfinite'] == 0
               and not case['finite_scalar']['official_pass_claim'] for case in finite['cases'])
    for name in ('native_conditional_queries', 'conditional_window_memory'):
        assert ref(HERE/(name+'.py'))['sha256'] == limit['sources'][name]['sha256']
    assert query['adapter']['sha256'] == limit['sources']['native_conditional_queries']['sha256']
    archived = {}
    for name in ('queries', 'memory-limit', 'memory-finite'):
        path = HERE/('native-context-'+name)/'preserved-artifacts/manifest.remote.json'
        item = json.loads(path.read_bytes())
        assert not item['local_archive_copy_verified']
        archived[name] = dict(manifest=ref(path), remote_archive=item['archive'],
            remote_file_count=len(item['files']), remote_file_bytes=sum(f['bytes'] for f in item['files']),
            local_archive_copy_verified=False,
            local_copy_scope='Metadata/source/results are saved locally. Tensor archive is complete on the original host; interrupted large SCP copies are not claimed verified.')
    observation_path = max((HERE/'conditional-memory-finite-observations').glob('[0-9]*.json'),
                           key=lambda p:int(p.stem))
    observation = json.loads(observation_path.read_bytes())
    assert not observation['driver_alive']
    output = REPO/'experiments/rl/results_conditional_memory_20261009.json'
    payload = dict(status='conditional_memory_owner_composition_verified_in_isolation_not_full_DT_repair',
        method=ref(REPO/'experiments/rl/PLAN.md'),
        original_collection=ref(REPO/'experiments/rl/results_single_background_completed_20261009.json'),
        existing_native_API_evidence=ref(REPO/'experiments/rl/results_native_context_readouts_20261009.json'),
        original_test=limit['original_test'], original_readout=limit['original_readout'],
        sources=limit['sources'],
        original_dtype_checks=dict(query_composition=ref(query_path), query_checks_passed=10,
            finite_derivative_limit=ref(limit_path), limit_checks_passed=28,
            unchanged_original_output_checks_passed=26, original_thresholds_changed=False),
        nonzero_local_identity=dict(result=ref(finite_path),
            scope=finite['limits'], finite_scalar_official_tolerance=None,
            cases=[dict(dtype=case['dtype'], positions_per_trajectory=case['sources_per_trajectory'],
                        nonzero_reference_effects=case['nonzero_reference_effects'],
                        finite_scalar=case['finite_scalar'], seconds=case['seconds'])
                   for case in finite['cases']]),
        lowering=ref(HERE/'conditional-window-lowering.json'),
        preservation=archived,
        launch_versions={name:ref(HERE/(name+'-launch.json')) for name in
                         ('native-conditional-queries','conditional-memory-limit','conditional-memory-finite')},
        observed_resources=dict(scope='Only the isolated native operator diagnostics, not model/DT capacity.',
            query_peak_allocated=query['peak_allocated_bytes'],
            limit_peak_allocated=limit['peak_allocated_bytes'],
            finite_peak_allocated=finite['peak_allocated_bytes'],
            finite_peak_reserved=finite['peak_reserved_bytes']),
        next_owner_seam='Use the original causal-conv interface to generate the selected hidden-row conditional q/k/v window; preserve its actual dtype, both finite endpoint orders, native norm/gate/projection owners and AppWorld capture release. Then measure combined tile residency/cost and unchanged frozen author cumulative deletion/RISE/MAS.',
        limits='Joint-background error has collection evidence, but GDN dominance and whole-model repair are not established. No top-k refinement, scalar correction, clipping or reference-model calls enter training.',
        observation=ref(observation_path), formal_training='TextCraft and AppWorld stopped',
        PPO_debug='Previously preserved; the original PPO NaN is not claimed reproduced or repaired.',
        QVA_modified=False, PPO_modified=False, production_modified=False,
        training_restart=False, checkpoint_restore=False,
        whole_DT_credit_repair_accepted=False, recorder=ref(Path(__file__)))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    path = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(path.read_bytes())
    current['observed_unix'] = observation['unix']
    current['observed_utc'] = datetime.fromtimestamp(observation['unix'], timezone.utc).isoformat()
    current['latest_conditional_memory_20261009'] = dict(status=payload['status'],
        receipt=ref(output), production_modified=False, whole_DT_credit_repair_accepted=False)
    current['latest_readonly_observation'] = dict(textcraft='Formal training stopped',
        appworld='Formal training stopped', diagnostic=payload['status'], receipt=ref(output))
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(receipt=str(output), query_checks=10, limit_checks=28,
        original_o_checks=26, finite_RMS=[c['finite_scalar']['normalized_RMS'] for c in finite['cases']],
        whole_DT_credit_repair_accepted=False)))


if __name__ == '__main__':
    main()
