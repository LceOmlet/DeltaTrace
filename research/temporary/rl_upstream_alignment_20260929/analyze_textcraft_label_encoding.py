"""Describe paired label readouts; no model calls, training changes or gates."""
import argparse
import json
import math
from pathlib import Path
import statistics

from analyze_textcraft_native_readout import source, describe, pearson


ROOT = Path(__file__).parent / 'textcraft-degradation-20261005'
LOCAL = ROOT / 'label-encoding-20261006/v2'
PRIOR = ROOT / 'readout-quality-20261006/v2'
INPUT = PRIOR / 'original-first-response-cases.json'
BOUNDARY = ROOT / 'native-minibatch-v4/native-minibatch-update-boundary.json'


def coordinate(row):
    return row['traj_uid'], row['source_step']


def summarize_pool(rows):
    outcomes = [row['observed_return'] for row in rows]
    mapping = {}
    for name in ('original', 'swapped'):
        variants = {}
        for variant in ('factual', 'full_eos'):
            probabilities = [row[name][variant + '_success_probability'] for row in rows]
            variants[variant] = dict(success_probability=describe(probabilities),
                brier=statistics.mean((p - g) ** 2 for p, g in zip(probabilities, outcomes)),
                pearson_actual_G=pearson(probabilities, outcomes))
        mapping[name] = dict(variants=variants,
            success_log_ratio=describe([row[name]['success_log_ratio'] for row in rows]),
            observed_target_log_ratio=describe([row[name]['observed_target_log_ratio'] for row in rows]))
    opposite = [row for row in rows if row['original']['success_log_ratio'] * row['swapped']['success_log_ratio'] < 0]
    return dict(n=len(rows), actual_success_fraction=statistics.mean(outcomes), mappings=mapping,
        paired_swapped_minus_original=dict(
            factual_success_probability=describe([row['probability_shift']['factual'] for row in rows]),
            full_eos_success_probability=describe([row['probability_shift']['full_eos'] for row in rows]),
            success_log_ratio=describe([row['success_log_ratio_shift'] for row in rows])),
        success_log_ratio_sign_flips=dict(count=len(opposite), denominator=len(rows),
            identities=[dict(traj_uid=row['traj_uid'], source_step=row['source_step']) for row in opposite],
            scope='Opposite strictly nonzero signs under two semantic label mappings; not token advantages.'))


def build_analysis(local, original_input, prior, boundary):
    paths = [local / f'rank{rank}-readout.json' for rank in range(2)]
    completion_path = local / 'completed.json'
    runs = [json.loads(path.read_bytes()) for path in paths]
    completed = json.loads(completion_path.read_bytes())
    assert all(run['phase'] == 'complete_native_readout' for run in runs)
    assert [len(run['cases']) for run in runs] == [32, 32]
    assert all(run['optimizer_steps'] == run['scheduler_steps'] == run['backward_calls'] ==
               run['finite_trace_calls'] == 0 for run in runs)
    assert [run['native_forward_calls'] for run in runs] == [32, 32]
    inputs = json.loads(original_input.read_bytes())
    inputs_by_key = {coordinate(row): row for rank in inputs['rank_cases'] for row in rank}
    assert len(inputs_by_key) == 64
    assert all(run['input_sha256'] == source(original_input)['sha256'] for run in runs)
    assert all(run['sampling'] == inputs['sampling'] and run['max_steps'] == inputs['max_steps'] for run in runs)
    sources = [source(path) for path in paths] + [source(completion_path), source(original_input), source(Path(__file__)),
        source(Path(__file__).with_name('analyze_textcraft_native_readout.py'))]
    boundary_data = json.loads(boundary.read_bytes()) if boundary.is_file() else None
    groups_by_uid = {}
    if boundary_data is not None:
        assert len(boundary_data['traj_uid']) == len(boundary_data['uid'])
        for uid, group in zip(boundary_data['traj_uid'], boundary_data['uid']):
            assert uid not in groups_by_uid or groups_by_uid[uid] == group
            groups_by_uid[uid] = group
        sources.append(source(boundary))
    pairs = []
    for run in runs:
        for case in run['cases']:
            saved = inputs_by_key[coordinate(case)]
            assert case['source_step'] == 0
            for field in ('observed_return', 'source_start', 'source_end'):
                assert case[field] == saved[field]
            assert case['original_query_ids'] == saved['selected_input_ids'][saved['source_end']:-1]
            assert case['context_tokens'] == len(saved['selected_input_ids']) <= 32768
            assert case['native_input_shape'] == [4, len(saved['selected_input_ids']) - 1]
            assert run['outcome_token_ids'] == [15, 16]
            assert case['success_class_indices'] == [1, 1, 0, 0]
            changed = [i for i, (a, b) in enumerate(zip(case['original_query_ids'], case['swapped_query_ids'])) if a != b]
            assert len(case['original_query_ids']) == len(case['swapped_query_ids']) and len(changed) == 2
            assert changed == case['changed_query_positions']
            assert sorted((case['original_query_ids'][i], case['swapped_query_ids'][i]) for i in changed) == [(15, 16), (16, 15)]
            values = case['native_outcome_log_probs']
            assert len(values) == 4 and all(len(row) == 2 for row in values)
            assert all(math.isfinite(value) for row in values for value in row)
            target_index = saved['observed_class_index']
            assert case['observed_class_indices'] == [target_index, target_index, 1 - target_index, 1 - target_index]
            row = dict(traj_uid=case['traj_uid'], source_step=case['source_step'], rank=run['rank'],
                case_index=case['case_index'], task_group_uid=groups_by_uid.get(case['traj_uid']),
                observed_return=case['observed_return'], source_start=case['source_start'], source_end=case['source_end'],
                context_tokens=case['context_tokens'], native_input_shape=case['native_input_shape'],
                native_logits_dtype=case['native_logits_dtype'], changed_query_positions=changed)
            for name, offset, success_index, observed_index in (
                    ('original', 0, 1, target_index), ('swapped', 2, 0, 1 - target_index)):
                row[name] = dict(success_class_index=success_index, observed_class_index=observed_index,
                    factual_success_probability=math.exp(values[offset][success_index]),
                    full_eos_success_probability=math.exp(values[offset + 1][success_index]),
                    success_log_ratio=values[offset][success_index] - values[offset + 1][success_index],
                    observed_target_log_ratio=values[offset][observed_index] - values[offset + 1][observed_index],
                    factual_log_probs=values[offset], full_eos_log_probs=values[offset + 1])
            row['probability_shift'] = {name: row['swapped'][name + '_success_probability'] - row['original'][name + '_success_probability']
                                        for name in ('factual', 'full_eos')}
            row['success_log_ratio_shift'] = row['swapped']['success_log_ratio'] - row['original']['success_log_ratio']
            pairs.append(row)
    assert len({coordinate(row) for row in pairs}) == 64 and {coordinate(row) for row in pairs} == set(inputs_by_key)
    successes = [row for row in pairs if row['observed_return'] == 1]
    failures = [row for row in pairs if row['observed_return'] == 0]
    assert len(successes) + len(failures) == 64
    group_summary = []
    for uid in sorted({row['task_group_uid'] for row in pairs if row['task_group_uid'] is not None}):
        rows = [row for row in pairs if row['task_group_uid'] == uid]
        group_summary.append(dict(task_group_uid=uid, **summarize_pool(rows)))
    previous_paths = [prior / f'rank{rank}-readout.json' for rank in range(2)]
    previous = {}
    for path in previous_paths:
        if path.is_file():
            old_run = json.loads(path.read_bytes())
            assert old_run['phase'] == 'complete_native_readout'
            assert old_run['input_sha256'] == source(original_input)['sha256']
            for row in old_run['cases']:
                assert coordinate(row) not in previous
                previous[coordinate(row)] = row
            sources.append(source(path))
    comparisons = []
    for row in pairs:
        old = previous.get(coordinate(row))
        if old is None:
            continue
        assert old['observed_return'] == row['observed_return']
        assert (old['source_start'], old['source_end'], old['context_tokens']) == (row['source_start'], row['source_end'], row['context_tokens'])
        differences = {variant: [row['original'][variant + '_log_probs'][index] - old['native_outcome_log_probs'][offset][index]
                                for index in range(2)] for variant, offset in [('factual', 0), ('full_eos', 1)]}
        comparisons.append(dict(traj_uid=row['traj_uid'], source_step=row['source_step'],
            factual_lp_difference_by_physical_class=differences['factual'], full_eos_lp_difference_by_physical_class=differences['full_eos'],
            success_log_ratio_difference=differences['factual'][1] - differences['full_eos'][1]))
    prior_summary = dict(n=len(comparisons), missing_identities=[dict(traj_uid=row['traj_uid'], source_step=row['source_step'])
        for row in pairs if coordinate(row) not in previous],
        scope='Same original literal prefix and physical 01 labels. Prior B4 contains two single deletions; current B4 contains two swapped-query rows. Differences are descriptive, not tolerance tests.', pairs=comparisons)
    if comparisons:
        prior_summary.update(factual_lp_difference={str(index): describe([row['factual_lp_difference_by_physical_class'][index] for row in comparisons]) for index in range(2)},
            full_eos_lp_difference={str(index): describe([row['full_eos_lp_difference_by_physical_class'][index] for row in comparisons]) for index in range(2)},
            success_log_ratio_difference=describe([row['success_log_ratio_difference'] for row in comparisons]))
    return dict(scope=__doc__, status='paired_label_sensitivity_observation_only', sources=sources,
        checkpoint=runs[0]['checkpoint'], selection=runs[0]['selection'], deployed_sources=[run['sources'] for run in runs],
        calls=dict(native_forwards=sum(run['native_forward_calls'] for run in runs), finite_attributions=0, backward=0, optimizer=0, scheduler=0),
        cases=64, observed_successes=len(successes), observed_failures=len(failures), task_groups=len(group_summary),
        missing_group_uids=[row['traj_uid'] for row in pairs if row['task_group_uid'] is None],
        group_scope='Original n8 uid joined to traj_uid from the native optimizer boundary; no reconstructed group assignment.',
        shape_budget=dict(native_B4=True, original_context_tokens=describe([row['context_tokens'] for row in pairs]),
            maximum_context_limit=32768, equal_length_query_bijection=True, only_two_label_ids_changed=True),
        sampling=inputs['sampling'], max_steps=inputs['max_steps'],
        runtime_dtypes=[dict(model=run['model_dtype'], native_fla_fp16=run['native_fla_fp16'], attention=run['event_attention_backend'],
            categorical_log_prob=sorted({case['native_logits_dtype'] for case in run['cases']})) for run in runs],
        pools=dict(all=summarize_pool(pairs), observed_success=summarize_pool(successes), observed_failure=summarize_pool(failures)),
        task_group_pools=group_summary, previous_native_readout_original_encoding=prior_summary,
        cases_by_uid_source_step=pairs, completed_unix=completed['completed_unix'],
        limitations=[
            'One checkpoint, 64 trajectories from eight original task groups; trajectories within a task group are dependent.',
            'The bijection changes two query label tokens, not values/meanings order, actual G or literal trajectory tokens.',
            'Factual and full-response-EOS success probabilities are model readouts, not environment counterfactual truth.',
            'A whole-response endpoint log ratio is not a per-token d or token advantage.',
            'Label sensitivity alone does not identify the complete cause of weak task gradients or historical degradation.',
            'No proposed repair, averaging, advantage scaling, numerical tolerance or learning module follows from these descriptive statistics.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, default=LOCAL)
    parser.add_argument('--original-input', type=Path, default=INPUT)
    parser.add_argument('--previous-dir', type=Path, default=PRIOR)
    parser.add_argument('--boundary', type=Path, default=BOUNDARY)
    args = parser.parse_args()
    result = build_analysis(args.local_dir, args.original_input, args.previous_dir, args.boundary)
    path = args.local_dir / 'label-encoding-analysis.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=path.as_posix(), sha256=source(path)['sha256'], cases=result['cases'],
        successes=result['observed_successes'], failures=result['observed_failures'], task_groups=result['task_groups'], pools=result['pools']), indent=2))
