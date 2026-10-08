"""Bind the already measured frozen sources for passive layer diagnostics.

No new sampling, estimator, candidate, model call or production patch. The
uniform four-source sample and the complete predicted-tail census retain
separate identities even where they refer to the same physical token.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main():
    corpus = json.loads((HERE / 'corpus.json').read_bytes())
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    groups = {uid: group for task in manifest['tasks'].values()
              for group in task['groups'] for uid in group['trajectory_uids']}
    uid_task = {row['traj_uid']: task for task, data in corpus['tasks'].items()
                for row in data['records']}
    points = {task: {} for task in corpus['tasks']}
    observations = []

    def add(task, point, cohort):
        uid, slot = point['traj_uid'], point['packed_slot']
        assert groups[uid]['split'] == 'development'
        key = (uid, slot)
        current = points[task].setdefault(key, dict(point, cohorts=[]))
        # Preserve two measured comparisons if the same token was queried in
        # two native batch contexts; do not average or silently choose one.
        current.setdefault('previous_comparisons', []).append(dict(point, cohort=cohort))
        if cohort not in current['cohorts']:
            current['cohorts'].append(cohort)

    for rank in (0, 1):
        path = HERE / 'author-collection-observations' / f'rank{rank}.json'
        observations.append(ref(path))
        value = json.loads(path.read_bytes())
        assert value['phase'] == 'complete'
        for batch in value['batches']:
            for row in batch['trajectories']:
                for point in row['uniform_deletions']:
                    add(batch['task'], dict(point, traj_uid=row['traj_uid']), 'uniform')
        for point in value['tail_results']:
            add(uid_task[point['traj_uid']], point, 'predicted_tail_census')

    plan = dict(scope=__doc__, frozen_manifest=ref(HERE / 'manifest.json'),
        frozen_corpus=ref(HERE / 'corpus.json'), native_comparisons=observations,
        operations=dict(model=0, DT=0, backward=0, optimizer=0, rollout=0),
        diagnostic_definition={
            'boundary': 's_l(i)=original joint finite coefficient at boundary l contracted with native factual-minus-single-EOS hidden states at the same boundary',
            'decoder_residual': 's_l(i)-s_(l+1)(i); includes the entire decoder, not a GDN-only or kernel-specific cause',
            'output_residual': 's_32(i)-native target log-prob difference; includes final norm and output head',
            'closure': 's_0 minus this same forward native deletion d telescopes into the decoder and output residuals. Record fresh_DT_d minus s_0 separately; compare fresh DT/native results with saved results separately. No drift correction or acceptance tolerance.',
            'ownership': 'No observer argument: passively retain existing coefficients/endpoint rows and call the unchanged native model and original _token_effect. Do not substitute any returned tensor.',
            'aggregation': 'Task, state, exposure, sampling cohort and crossed predicted/native ratio bins remain separate. Report conditional distributions and state-level frequencies, without pooled raw heavy-tail means.',
            'quality': 'Original author cumulative deletion, RISE and MAS remain method-quality evidence. Layer contractions are only localization diagnostics.'}, tasks={})
    for task, by_key in points.items():
        records = {r['traj_uid']: r for r in corpus['tasks'][task]['records']}
        files = {r['sha256']: r for r in corpus['tasks'][task]['native_files']}
        entries = []
        for uid in sorted({key[0] for key in by_key}):
            record = records[uid]
            occurrence = record['native_occurrences'][0]
            group = groups[uid]
            selected = sorted((p for (u, _), p in by_key.items() if u == uid),
                              key=lambda p: p['packed_slot'])
            entries.append(dict(traj_uid=uid, initial_state_sha256=group['initial_state_sha256'],
                previously_examined=group['previously_examined'],
                first_stage=uid in group['first_stage_uids'],
                native=files[occurrence['file_sha256']], occurrence=occurrence,
                selected_tokens=occurrence['selected_tokens'], queries=selected))
        # This is the existing readout's length ordering for B4, not a new
        # sample selection. Width/batch drift must be recorded, not corrected.
        entries.sort(key=lambda e: (e['selected_tokens'], e['traj_uid']))
        batches = [entries[start:start+4] for start in range(0, len(entries), 4)]
        plan['tasks'][task] = dict(entries=entries, batches=[dict(
            uids=[e['traj_uid'] for e in batch], actual_rows=len(batch),
            B4_identity_controls=4-len(batch), padded_width=max(e['selected_tokens'] for e in batch),
            native_paired_forwards=max(len(e['queries']) for e in batch)) for batch in batches],
            unique_queries=len(by_key), uniform=sum('uniform' in p['cohorts'] for p in by_key.values()),
            tail_census=sum('predicted_tail_census' in p['cohorts'] for p in by_key.values()))
        assert plan['tasks'][task]['uniform'] == 128
        assert plan['tasks'][task]['tail_census'] == 37
    (HERE / 'layer-collection-inputs.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({t: dict(trajectories=len(d['entries']), unique_queries=d['unique_queries'],
        DT_B4_calls=len(d['batches']), native_paired_forwards=sum(b['native_paired_forwards'] for b in d['batches']),
        max_width=max(b['padded_width'] for b in d['batches'])) for t, d in plan['tasks'].items()}))


if __name__ == '__main__':
    main()
