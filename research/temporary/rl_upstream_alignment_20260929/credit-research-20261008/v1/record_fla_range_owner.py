"""Bind the numerical repair's exact evidence without claiming credit repair."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    official = HERE/'range-owner-replay/exact-inputs/result.json'
    tests = json.loads(official.read_bytes())
    assert len(tests['cases']) == 2
    assert all(case['status']=='passed' and len(case['checks'])==11 for case in tests['cases'])
    folder = HERE/'range-owner-replay-observations'
    observation = max((p for p in folder.glob('*.json') if p.stem.isdecimal()), key=lambda p:int(p.stem))
    snapshot = json.loads(observation.read_bytes())
    assert not snapshot['driver_alive']
    ranks = [json.loads((folder/f'rank{rank}.json').read_bytes()) for rank in (0, 1)]
    assert all(r['phase']=='complete' and r['operations']['DT']==1 and r['operations']['optimizer']==0 for r in ranks)
    manifest = json.loads((HERE/'range-owner-replay/preserved-artifacts/manifest.json').read_bytes())
    extra = json.loads((HERE/'range-owner-replay/exact-inputs/manifest.json').read_bytes())
    assert manifest['all_file_SHA256_verified'] and extra['all_SHA256_verified']
    result = dict(status='numerical_boundary_repair_verified_in_isolated_original_B4_replay',
        scope='Original AppWorld FP16 cotangent overflow; not repaired extreme attribution, whole-method faithfulness or original PPO NaN.',
        official_checks=dict(receipt=ref(official), original_test=tests['original_test'],
            cases=tests['cases'], checks_passed=22, checks_failed=0,
            reference_input_precision=tests['reference_input_precision'],
            reference_seed_precision=tests['reference_seed_precision'],
            test_scope='Real failing 64-token block, all B4 x 8 heads, nonzero initial state; native and coincident finite adjoints. No ht/dh0 or nonzero finite oracle claim.'),
        repair=dict(owner_sha256='33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd',
            original_AppWorld_owner_sha256='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72',
            change='Minimum integer power-of-two cotangent representation before native FP16 conversion; every returned coefficient restored. Original FLA, allocation rule and call count unchanged.',
            unchanged_head_check=ref(HERE/'fla-seed-range-unchanged-heads.json'),
            prepared_owner=ref(HERE/'native-fla-range-owner-prepared/qwen35_gdn_finite.py'),
            preserves_AppWorld_consumed_capture_release=True),
        replay=dict(launch=ref(HERE/'range-owner-replay-launch.json'), observation=ref(observation),
            ranks=[dict(rank=r['rank'], phase=r['phase'], operations=r['operations'],
                        elapsed_seconds=r['elapsed_seconds'], DT_rounds=r['batches'][0]['rounds'],
                        finite_runner=r['finite_runner'], isolated_range_owner=r['isolated_range_owner']) for r in ranks],
            limits='Timing includes preparation and cold work; no speedup claim. Endpoint phase memory is not a peak measurement.'),
        debug_preservation=dict(manifest=ref(HERE/'range-owner-replay/preserved-artifacts/manifest.json'),
            files=len(manifest['files']), bytes=sum(v['bytes'] for v in manifest['files']), archive=manifest['archive'],
            exact_inputs_and_original_test=ref(HERE/'range-owner-replay/exact-inputs/manifest.json'),
            all_file_SHA256_verified=True),
        missing_query_continuation=dict(launch=ref(HERE/'single-background-range-resume-launch.json'),
            reused_original_points=74, reused_replay_points=8, remaining_queries=83,
            expected_total_queries=165, expected_new_B4_DT_calls_per_rank=20,
            status='running_diagnostic_at_recording'),
        QVA_modified=False, PPO_modified=False, credit_clipping_or_correction=False,
        production_modified=False, formal_restart=False, extreme_credit_repaired=False,
        original_PPO_NaN_fixed=False, source=ref(Path(__file__)))
    output = REPO/'experiments/rl/results_fla_range_owner_20261009.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    path = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(path.read_bytes())
    current['latest_fla_range_owner_20261009'] = dict(receipt=ref(output), status=result['status'],
        numerical_repair_owner_sha256=result['repair']['owner_sha256'],
        production_modified=False, formal_restart=False, extreme_credit_repaired=False,
        continuation=json.loads((HERE/'single-background-range-resume-launch.json').read_bytes()))
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(receipt=str(output), checks_passed=22, replay_ranks=[r['phase'] for r in ranks])))


if __name__ == '__main__':
    main()
