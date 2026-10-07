"""Describe the owner's measured curve without reimplementing RISE/MAS."""
import csv
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]


def receipt(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def main():
    destination = HERE / 'actual-results'
    transport = json.loads((destination / 'transport.json').read_bytes())
    for item in transport:
        assert receipt(Path(item['local_path']))['sha256'] == item['sha256']
    ranks = [json.loads((destination / f'results/rank{rank}.json').read_bytes()) for rank in (0, 1)]
    assert all(rank['phase'] == 'complete' for rank in ranks)
    assert ranks[0]['views'] == ranks[1]['views'] or all(
        ranks[0]['views'][name]['author_arrays'] == ranks[1]['views'][name]['author_arrays']
        and ranks[0]['views'][name]['author_return'] == ranks[1]['views'][name]['author_return']
        and [p['logp'] for p in ranks[0]['views'][name]['score_points']] == [p['logp'] for p in ranks[1]['views'][name]['score_points']]
        for name in ranks[0]['views'])
    original = AUDIT / 'direct-target-prefix-runtime-20261007/v1/first-native-artifacts/rank1-readout-native-batch-6.pt'
    assert receipt(original)['sha256'] == ranks[0]['native_sha256']
    native = torch.load(original, map_location='cpu', weights_only=False)
    candidate = ranks[0]['geometry']['candidate']
    row = candidate['row']
    positions = ranks[0]['source_positions']
    signed = native['native_signed'][row].double()
    views = {}
    csv_rows = []
    for name, view in ranks[0]['views'].items():
        points = view['score_points']
        groups = []
        previous = set()
        for index, point in enumerate(points):
            changed = set(point['changed_input_positions'])
            assert previous <= changed
            added = sorted(changed - previous)
            delta = signed[added]
            group = dict(step=index, changed_positions=len(changed), newly_changed_positions=len(added),
                original_DT_sum_of_newly_changed=float(delta.sum()),
                original_DT_negative_count=int((delta < 0).sum()),
                original_DT_positive_count=int((delta > 0).sum()),
                original_DT_zero_count=int((delta == 0).sum()),
                native_logp=point['logp'],
                native_kept_minus_deleted_for_step=None if index == 0 else points[index - 1]['logp'] - point['logp'],
                candidate_changed=candidate['packed_slot'] in changed,
                seconds=point['seconds'])
            groups.append(group)
            csv_rows.append(dict(view=name, **group))
            previous = changed
        arrays = view['author_arrays']
        increases = [dict(from_step=i - 1, to_step=i,
            logp_increase=points[i]['logp'] - points[i - 1]['logp'])
            for i in range(1, len(points)) if points[i]['logp'] > points[i - 1]['logp']]
        views[name] = dict(author_return=dict(zip(view['author_fields'], view['author_return'])),
            author_arrays=arrays, groups=groups,
            actual_logp_increases_on_deletion=increases,
            candidate_first_changed_step=next(g['step'] for g in groups if g['candidate_changed']),
            last_step=groups[-1],
            native_forward_seconds=sum(point['seconds'] for point in points),
            note='Groups are observed changed IDs, not reconstructed author sorting. Original EOS source slots may remain unchanged. Native step effects are context-dependent group effects, not individual-token ground truth.')
    previous = json.loads((AUDIT / 'direct-target-reference-interaction-20261007/v1/analysis.json').read_bytes())
    controls = dict(ranks_same_author_arrays_and_scores=True,
        maximum_twin_difference=max(abs(x) for rank in ranks for view in rank['views'].values() for p in view['score_points'] for x in p['twin_row_score_differences']),
        other_three_row_scores_constant=all(len({tuple(p['other_three_joint_logp']) for view in rank['views'].values() for p in view['score_points']}) == 1 for rank in ranks),
        factual_matches_previous_native_endpoint=all(v['groups'][0]['native_logp'] == previous['endpoints']['F_factual_all_sources'] for v in views.values()),
        all_EOS_matches_previous_native_endpoint=all(v['groups'][-1]['native_logp'] == previous['endpoints']['B_all_prior_sources_EOS'] for v in views.values()),
        local_LoRA_B_shards_zero=all(not shard['nonzero'] for rank in ranks for shard in rank['lora_B_local_shards']))
    launch = json.loads((HERE / 'launch.json').read_bytes())
    out = dict(scope='One original real AppWorld trajectory. Original author guided deletion function with its default k=20; original joint action Y supplied through a literal-ID/scalar-score adapter. Not the pristine single-suffix evaluator or a paper-scale quality result.',
        source_sha256=ranks[0]['source_sha256'], native_sha256=ranks[0]['native_sha256'], candidate=candidate,
        source_count=len(positions), author=ranks[0]['author'], views=views, controls=controls,
        measurements=dict(native_forwards_per_rank=42, DT_calls=0, rollout_calls=0, backward=0, optimizer=0,
            elapsed_from_driver_launch_to_last_worker_complete_seconds=max(r['unix'] for r in ranks) - launch['launched_unix'],
            worker_peak_observed_pss_bytes=[max(p['pss_bytes'] for v in r['views'].values() for p in v['score_points']) for r in ranks]),
        interpretation='High positive-ranked sources remove much of the joint target log-prob. The final signed-ranked group contains only negative DT values among changed IDs, yet deleting this group decreases native log-prob. This measured group-context mismatch complements the single-token sign mismatch; it does not establish all DT attributions are poor, decompose error per token, or supply an FA/FLA tolerance threshold.',
        evaluation_only='The positive-only MAS view is passed only to the unmodified author metric; it never changes raw DT, Q/V/A, whitening or PPO. Raw nonmonotonic scores are retained alongside the author normalized/penalized arrays.',
        author_metric_scope='The author normalizes and takes a running minimum in normalized_model_response. Its corrected_scores are evaluation-only. Neither transformation is applied to credits. No copied metric, new threshold, or pass/fail label.',
        sources=[receipt(HERE / 'owner-source/ft_ifr_improve.py'),receipt(destination / 'transport.json'),receipt(original),receipt(AUDIT / 'direct-target-reference-interaction-20261007/v1/analysis.json')],
        active_formal_state='TextCraft original first update held; AppWorld formal terminal. No restore, formal restart, credit change or release.')
    (HERE / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (HERE / 'curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(controls=controls, measurements=out['measurements'],
        views={name: {k:v[k] for k in ('author_return','candidate_first_changed_step','last_step')} for name,v in views.items()})))


if __name__ == '__main__':
    main()
