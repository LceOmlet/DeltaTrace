"""Bind saved debugging and native convolution seam evidence; no repair claim."""
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
    result_path = HERE/'conditional-conv-windows-result.json'
    result = json.loads(result_path.read_bytes())
    launch_path = HERE/'conditional-conv-windows-launch.json'
    launch = json.loads(launch_path.read_bytes())
    observation_path = max(HERE.glob('conditional-conv-windows-observe-[0-9]*.json'),
                           key=lambda p:int(p.stem.rsplit('-',1)[-1]))
    observation = json.loads(observation_path.read_bytes())
    assert not observation['driver_alive'] and result['phase'] == 'complete' and result['passed']
    assert len(result['cases']) == 2
    assert all(c['B'] == 4 and c['source_count'] == 199 for c in result['cases'])
    assert not any(v for c in result['cases'] for v in c['composition_mismatches'].values())
    assert sum(len(c['reference_groups'])*2 for c in result['cases']) == 52
    assert all(all(g['owner_checks'].values()) for c in result['cases'] for g in c['reference_groups'])
    sources = {name:ref(HERE/name) for name in launch['script_sha256']}
    assert all(item['sha256'] == launch['script_sha256'][name] for name,item in sources.items())
    manifest_path = HERE/'native-context-conv-windows/preserved-artifacts/manifest.remote.json'
    manifest = json.loads(manifest_path.read_bytes())
    assert not manifest['local_archive_copy_verified']
    artifacts = {f['sha256'] for f in manifest['files']}
    assert all(c['exact_artifact']['sha256'] in artifacts for c in result['cases'])
    debug_path = REPO/'experiments/rl/results_preserved_actor_debug_20261008.json'
    debug = json.loads(debug_path.read_bytes())
    assert debug['files'] == 44 and debug['lossless_archive']['local_verified']
    assert Path(debug['lossless_archive']['local_path']).is_file()
    owner_path = HERE/'conditional-conv-owner.json'
    owner = json.loads(owner_path.read_bytes())
    assert owner['phase'] == 'complete' and all(owner['import_availability'].values())
    assert owner['sources']['causal_conv1d_fn']['file']['sha256'] == result['owner']['sha256']
    output = REPO/'experiments/rl/results_conditional_conv_20261009.json'
    payload = dict(status='native_conv_single_row_window_seam_verified_not_full_DT_repair',
        PPO_debug=dict(receipt=ref(debug_path), files=44, bytes=debug['bytes'],
            original_NaN_repaired=False, preservation_scope='Original receipt already verifies local lossless archive and per-file SHA; not another PPO replay.'),
        joint_background_evidence=ref(REPO/'experiments/rl/results_single_background_completed_20261009.json'),
        conditional_memory=ref(REPO/'experiments/rl/results_conditional_memory_20261009.json'),
        original_owners=ref(owner_path), sources=sources, launch=ref(launch_path),
        executed_base_commit=launch['base_commit'], result=ref(result_path),
        official_test=dict(**result['official_test'],
            url='https://github.com/Dao-AILab/causal-conv1d/blob/v1.5.0/tests/test_causal_conv1d.py',
            scope='Exact test_causal_conv1d dtype rtol/atol assignments extracted by AST; original public fn/ref on real operands. No full official test-suite or backward claim.'),
        operator_checks=dict(original_native_reference_forward_checks=52,
            composition='All valid four-lag preactivation, fused output and q/k/v bitwise equal to complete public convolution calls.',
            cases=[dict(dtype=c['dtype'], positions_per_trajectory=c['source_count'], trajectories=c['B'],
                actually_changed_rows=c['B']*c['source_count']-c['zero_change_rows'],
                original_tolerance=c['original_tolerance'], mismatches=c['composition_mismatches'],
                seconds_including_diagnostic_references=c['seconds'],
                peak_allocated_bytes=c['peak_allocated_bytes'], peak_reserved_bytes=c['peak_reserved_bytes'])
                for c in result['cases']],
            sample_scope='Historical real GDN operator captures only. Not a substitute for frozen TextCraft/AppWorld primary cumulative deletion/RISE/MAS.'),
        preservation=dict(manifest=ref(manifest_path), remote_archive=manifest['archive'],
            remote_files=len(manifest['files']), local_tensor_archive_verified=False,
            local_scope='Source, results, exact paths and hashes; new large tensors remain archived on original host.'),
        inspection_failure=dict(process=ref(HERE/'conditional-conv-owner-failed-recursive-glob-process.json'),
            source=ref(HERE/'conditional-conv-owner-failed-recursive-glob-inspector.py'),
            reason='Read-only recursive receipt glob blocked in NFS getattr; own process identity checked and stopped. Replaced only inspector scan; no runtime/package changes.'),
        observation=ref(observation_path), observed_host_available_bytes=observation['host']['available'],
        next='Integrate this original convolution window with conditional memory in an isolated GDN owner, stream native-chunk tiles, then test unchanged frozen author curves and separated tail cells. No default launch change before acceptance.',
        formal_training='TextCraft and AppWorld stopped', checkpoint_restore=False,
        QVA_modified=False, PPO_modified=False, production_modified=False,
        whole_DT_credit_repair_accepted=False)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    current_path = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(current_path.read_bytes())
    current.update(observed_unix=observation['unix'],
                   observed_utc=datetime.fromtimestamp(observation['unix'],timezone.utc).isoformat())
    current['latest_conditional_conv_20261009'] = dict(status=payload['status'], receipt=ref(output),
        executed_base_commit=launch['base_commit'], sources=sources, production_modified=False,
        whole_DT_credit_repair_accepted=False)
    current['latest_readonly_observation'] = dict(textcraft='Formal training stopped',
        appworld='Formal training stopped', diagnostic=payload['status'], receipt=ref(output))
    current_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=str(output), checks=52, whole_DT_repair=False)))


if __name__ == '__main__':
    main()
