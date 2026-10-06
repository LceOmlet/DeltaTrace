"""Stdlib review of real author-curve raw receipts; no metric reimplementation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def artifact(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def ids_hash(ids):
    return hashlib.sha256(struct.pack('<' + 'q' * len(ids), *ids)).hexdigest()


def summary(values):
    return dict(count=len(values), mean=math.fsum(values) / len(values),
                min=min(values), max=max(values), mean_abs=math.fsum(abs(v) for v in values) / len(values))


def review(directory, cases_path):
    inputs = json.loads(cases_path.read_bytes())
    cases = {c['traj_uid']: c for rows in inputs['rank_cases'] for c in rows if c['actual_G'] == 1.0}
    prepared = json.loads((directory / 'prepared.json').read_bytes())
    job = json.loads((directory / 'job.json').read_bytes())
    complete = json.loads((directory / 'completed.json').read_bytes())
    ranks = [json.loads((directory / f'rank{r}-curves.json').read_bytes()) for r in range(2)]
    primary, duplicates, arithmetic = {}, [], []
    for rank, data in enumerate(ranks):
        assert data['phase'] == 'complete_author_reward_curves'
        assert len(data['cases']) == 11 and data['native_forward_calls'] == 462
        assert all(data[key] == 0 for key in ('optimizer_steps', 'scheduler_steps', 'backward_calls', 'finite_trace_calls'))
        for row in data['cases']:
            case = cases[row['traj_uid']]
            contract = row['contract']
            assert contract['literal_input_sha256'] == ids_hash(case['selected_input_ids'])
            assert contract['literal_prefix_sha256'] == ids_hash(case['selected_input_ids'][:-1])
            assert contract['target_ids'] == [case['target_id']] and contract['target_eos_appended'] is False
            assert row['source_start'] == case['source_start'] and row['source_end'] == case['source_end']
            assert row['source_step'] == case['source_step'] and row['observed_return'] == case['observed_return'] == 1.0
            assert row['native_scoring_calls'] == 42
            n = case['source_end'] - case['source_start']
            for view, result in row['curves']['views'].items():
                points = result['score_points']
                assert len(points) == result['native_score_calls'] == 21
                assert len(result['author_return']) == 3
                assert all(len(array) == 21 for array in result['author_curve'].values())
                assert points[0]['author_deleted_source_indices'] == []
                assert len(points[-1]['author_deleted_source_indices']) == n
                assert set(points[-1]['author_deleted_source_indices']) == set(range(n))
                prior = []
                for index, point in enumerate(points):
                    deleted = point['author_deleted_source_indices']
                    assert point['point_index'] == index
                    assert deleted == prior + point['author_current_group_source_indices']
                    assert point['author_deleted_input_positions'] == [case['source_start'] + j for j in deleted]
                    actual_sum = math.fsum(case['saved_source_d_from_A'][j] for j in deleted)
                    arithmetic.append(abs(actual_sum - point['same_set_saved_d_sum']))
                    prior = deleted
            if row['traj_uid'] in primary:
                duplicates.append(dict(rank=rank, case_index=row['case_index'], traj_uid=row['traj_uid'],
                    factual_logp_difference={v: row['curves']['views'][v]['score_points'][0]['raw_logp'] -
                        primary[row['traj_uid']]['curves']['views'][v]['score_points'][0]['raw_logp']
                        for v in row['curves']['views']}))
            else:
                primary[row['traj_uid']] = row
    assert len(primary) == len(cases) == 21 and len(duplicates) == 1
    results = {}
    for view in ('positive_MAS', 'signed_RISE'):
        rows = [row['curves']['views'][view] for row in primary.values()]
        effects = [r['score_points'][0]['raw_logp'] - r['score_points'][-1]['raw_logp'] for r in rows]
        sums = [r['score_points'][-1]['same_set_saved_d_sum'] for r in rows]
        results[view] = dict(
            author_metrics={name: summary([r['author_return'][i] for r in rows])
                for i, name in enumerate(('rise', 'mas', 'rise_plus_ap'))},
            raw_full_source_native_logp_effect=summary(effects),
            saved_d_total=summary(sums), raw_endpoint_difference=summary([a-b for a, b in zip(sums, effects)]),
            raw_full_source_native_negative_count=sum(v < 0 for v in effects),
            full_source_saved_native_opposite_sign_count=sum(a*b < 0 for a, b in zip(sums, effects)),
            original_normalized_model_response_all_zero_count=sum(
                all(v == 0.0 for v in r['author_curve']['normalized_model_response']) for r in rows),
            original_normalized_model_response_all_zero_uids=[uid for uid, row in primary.items()
                if all(v == 0.0 for v in row['curves']['views'][view]['author_curve']['normalized_model_response'])])
    repeated_endpoints = {
        name: summary([abs(row['curves']['views']['positive_MAS']['score_points'][index]['raw_logp'] -
                            row['curves']['views']['signed_RISE']['score_points'][index]['raw_logp'])
                       for row in primary.values()])
        for name, index in (('factual', 0), ('all_source_eos', -1))}
    return dict(
        scope='Independent stdlib source/ID/count/cumulative-set and raw endpoint review. Original author metrics are read, never recomputed. No model/Torch/DT/backward/update.',
        sources=[artifact(directory / name) for name in
            ('prepared.json', 'job.json', 'completed.json', 'rank0-curves.json', 'rank1-curves.json',
             'submitted-source/verify_textcraft_author_reward_curve.py',
             'submitted-source/author_reward_curve_adapter.py',
             'submitted-source/stage_textcraft_author_reward_curve.py')] + [artifact(cases_path), artifact(Path(__file__))],
        source_binding_same={key: prepared[key] == job[key] for key in
            ('sources', 'diagnostic_sources', 'input', 'config_source', 'checkpoint')},
        actual_counts=dict(unique_success_first_responses=21, padded_slots=22, scoring_calls_per_rank=[462,462],
                          total_scoring_calls=924, unique_curve_points_per_view=441, duplicates=duplicates),
        cumulative_set_contract='Each exact author group extends the prior set, maps to the saved source span, and ends with all source positions; original EOS positions remain represented.',
        saved_subset_sum_recalculation_max_abs=max(arithmetic),
        original_author_metrics_and_raw_endpoints=results,
        same_input_two_view_endpoint_difference=repeated_endpoints,
        wall_seconds=complete['completed_unix']-job['started_unix'],
        limitations=[
            '21 saved G1 first responses at checkpoint25; G0 and later responses are outside this diagnostic.',
            'Ordering uses saved inverse d_from_A from stored FP32 A, not a fresh/pre-storage DT vector.',
            'Author normalized RISE/MAS cannot establish reward-effect amplitude accuracy or PPO task-gradient strength.',
            'Negative factual-minus-full-EOS endpoints can make the original normalized curve all zero; RISE=0 is not automatically evidence of quality.',
            'Cumulative-set native effects test these whole-response deletion curves, not per-token or world counterfactual correctness.',
            'Endpoint/duplicate differences are descriptive; no official tolerance or numerical pass gate is invented.',
            'No production/parameter change or training-success improvement is demonstrated.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--cases-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = review(args.input_dir, args.cases_path)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(args.output.resolve()), sha256=artifact(args.output)['sha256'],
                         counts=result['actual_counts'], summaries=result['original_author_metrics_and_raw_endpoints']), indent=2))
