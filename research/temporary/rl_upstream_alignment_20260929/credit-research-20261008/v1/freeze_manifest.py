"""Freeze collection-wide research splits, without inspecting credit magnitudes.

Split complete initial policy states as clusters. Copied UIDs count once.
Previously examined native batches and all their matching initial states stay
on the development side. Selection never reads d, A, success rank or length.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
SEED = '20261008-credit-research-v1'
EXPOSED_BATCHES = {
    '19b18c5f4204ec4d88e40d72d350a06954f64452824a8662e95497319f89a37a',
    'aaa03be7fb10624918e5289aa1fc3409ae2d81ce32243a84df5539c48727be85',
    '3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a',
}


def digest(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def order(*parts):
    return hashlib.sha256('\0'.join((SEED,)+parts).encode()).hexdigest()


def freeze(corpus):
    result = {'seed': SEED, 'scope': __doc__, 'tasks': {},
              'exposed_batch_sha256': sorted(EXPOSED_BATCHES),
              'evaluation_phase': 'No candidate comparison or test-side quality scoring performed',
              'metric_owner': {'function': 'ft_ifr_improve.faithfulness_test_skip_tokens',
                  'sha256': '583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1',
                  'k': 20, 'rise_view': 'signed d cast FP32',
                  'mas_view': 'd cast FP32 then clamp_min(0), evaluation only'},
              'aggregation': 'Per-environment equal-initial-state means of trajectory means, '
                  'with trajectory arithmetic means also reported; paired same-UID differences. '
                  'Report state-cluster uncertainty; no pooled task scalar, '
                  'token weighting, reward weighting or replacing missing scores with zero.'}
    for task, data in corpus['tasks'].items():
        groups = {}
        for row in data['records']:
            groups.setdefault(row['initial_state_sha256'], []).append(row)
        exposed = {key for key, rows in groups.items() if any(
            occurrence['file_sha256'] in EXPOSED_BATCHES
            for row in rows for occurrence in row['native_occurrences'])}
        target = len(groups)//2
        if len(exposed) > target:
            raise ValueError('Previously inspected groups leave insufficient frozen holdout')
        remaining = sorted(set(groups)-exposed, key=lambda group: order(task, group))
        dev = exposed | set(remaining[:target-len(exposed)])
        output_groups = []
        for key in sorted(groups):
            rows = sorted(groups[key], key=lambda row: order(task, row['traj_uid']))
            output_groups.append({
                'initial_state_sha256': key, 'split': 'development' if key in dev else 'test',
                'previously_examined': key in exposed,
                'trajectory_uids': [row['traj_uid'] for row in rows],
                'first_stage_uids': [row['traj_uid'] for row in rows[:2]],
                'missing_native_uids': [row['traj_uid'] for row in rows if not row['native_occurrences']],
            })
        counts = {}
        for split in ('development', 'test'):
            split_groups = [group for group in output_groups if group['split'] == split]
            uids = {uid for group in split_groups for uid in group['trajectory_uids']}
            records = [row for row in data['records'] if row['traj_uid'] in uids]
            lengths = [row['context_tokens'] for row in records]
            counts[split] = {
                'groups': len(split_groups), 'trajectories': len(uids),
                'first_stage_trajectories': sum(len(g['first_stage_uids']) for g in split_groups),
                'completed_native': sum(bool(row['native_occurrences']) for row in records),
                'missing_native': sum(not row['native_occurrences'] for row in records),
                'context_tokens': {'min': min(lengths), 'median': statistics.median(lengths),
                                   'max': max(lengths)},
                'reward_range': [min(row['reward'] for row in records), max(row['reward'] for row in records)],
                # B4 geometry stays B4; duplicated tail rows do not become observations.
                'first_stage_B4_requests': math.ceil(sum(len(g['first_stage_uids']) for g in split_groups)/4),
            }
        result['tasks'][task] = {'source': data['source'], 'counts': counts, 'groups': output_groups}
    return result


if __name__ == '__main__':
    path = HERE/'corpus.json'
    result = freeze(json.loads(path.read_bytes()))
    result['corpus'] = digest(path)
    result['builder'] = digest(Path(__file__).resolve())
    output = HERE/'manifest.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'manifest': digest(output), 'tasks': {
        task: data['counts'] for task, data in result['tasks'].items()}}, indent=2))
