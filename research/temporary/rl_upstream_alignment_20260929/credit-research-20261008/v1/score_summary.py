"""Aggregate unmodified author per-trajectory metrics on frozen identities.

No RISE/MAS definition, curve normalization or numerical tolerance lives here.
Report each environment separately; duplicate rank/copy rows are rejected.
Missing/nonfinite owner results are explicit, never zero or silent successes.
"""
import math
import statistics


def summarize(manifest, rows, split):
    expected = {}
    for task, data in manifest['tasks'].items():
        for group in data['groups']:
            if group['split'] == split:
                for uid in group['trajectory_uids']:
                    expected[(task, uid)] = group['initial_state_sha256']
    values = {}
    for row in rows:
        identity = row['task'], row['traj_uid']
        if identity not in expected:
            raise ValueError('Scored identity is outside this frozen split')
        if identity in values:
            raise ValueError('Duplicate rank/copied trajectory cannot receive a second weight')
        values[identity] = row
    output = {}
    for task in manifest['tasks']:
        identities = [identity for identity in expected if identity[0] == task]
        metrics = {}
        for metric in ('rise', 'mas'):
            valid, by_state = [], {}
            nonfinite, missing = [], []
            for identity in identities:
                value = values.get(identity, {}).get('metrics', {}).get(metric)
                if value is None:
                    missing.append(identity[1])
                elif not math.isfinite(value):
                    nonfinite.append(identity[1])
                else:
                    valid.append(float(value))
                    by_state.setdefault(expected[identity], []).append(float(value))
            group_means = [statistics.fmean(group) for group in by_state.values()]
            metrics[metric] = {
                'expected_trajectories': len(identities), 'scored_finite': len(valid),
                'missing_uids': missing, 'nonfinite_uids': nonfinite,
                'complete': len(valid) == len(identities),
                'available_trajectory_mean': statistics.fmean(valid) if valid else None,
                'available_equal_state_mean': statistics.fmean(group_means) if group_means else None,
                'scored_state_groups': len(group_means),
                'state_mean_standard_error': statistics.stdev(group_means)/math.sqrt(len(group_means))
                    if len(group_means) > 1 else None,
            }
        output[task] = metrics
    return output


def paired_differences(manifest, baseline, candidate, split):
    def indexed(rows):
        result = {}
        for row in rows:
            key = row['task'], row['traj_uid']
            if key in result:
                raise ValueError('Duplicate comparison identity')
            result[key] = row
        return result
    left, right = indexed(baseline), indexed(candidate)
    # Validate original identities, even for unmatched rows.
    summarize(manifest, baseline, split)
    summarize(manifest, candidate, split)
    paired = []
    for key in sorted(left.keys() & right.keys()):
        metrics = {}
        for name in ('rise', 'mas'):
            first = left[key].get('metrics', {}).get(name)
            second = right[key].get('metrics', {}).get(name)
            metrics[name] = second-first if first is not None and second is not None else None
        paired.append({'task': key[0], 'traj_uid': key[1], 'metrics': metrics})
    return summarize(manifest, paired, split)
