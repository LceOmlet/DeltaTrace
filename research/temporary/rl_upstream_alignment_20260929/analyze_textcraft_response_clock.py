"""Independently describe saved response-clock readouts with stdlib only.

Readouts are whole-response factual/EOS endpoint log probabilities, not token
credits. No model, tokenizer, Torch, rollout, DT, backward or optimizer is used.
All comparisons are descriptive; no probability or numerical pass threshold is
defined. Only the requested output JSON is written when the CLI is executed.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT / 'textcraft-degradation-20261005'
LOCAL = ROOT / 'response-clock-20261006/v2'
INPUT = ROOT / 'readout-quality-20261006/v2/original-first-response-cases.json'
BOUNDARY = ROOT / 'native-minibatch-v4/native-minibatch-update-boundary.json'
PREVIOUS = ROOT / 'label-encoding-20261006/v2'
DRIVER = AUDIT / 'verify_textcraft_response_clock.py'
ZERO_OPERATIONS = ('optimizer_steps', 'scheduler_steps', 'backward_calls', 'finite_trace_calls')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def read(path):
    return json.loads(path.read_bytes())


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def coordinate(row):
    return row['traj_uid'], row['source_step']


def describe(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, median=None, min=None, max=None, abs_mean=None,
                    rms=None, std_population=None, positive=0, negative=0, zero=0, quantiles={})
    ordered = sorted(values)
    quantiles = {}
    for q in (.1, .25, .5, .75, .9, .95):
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        quantiles[str(q)] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), abs_mean=statistics.mean(abs(v) for v in values),
                rms=math.sqrt(statistics.mean(v * v for v in values)),
                std_population=statistics.pstdev(values), positive=sum(v > 0 for v in values),
                negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values), quantiles=quantiles)


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    numerator = math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    denominator = math.sqrt(math.fsum((x - xm) ** 2 for x in xs) * math.fsum((y - ym) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def replacements_from_source(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    declarations = [node for node in tree.body if isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == 'REPLACEMENTS' for target in node.targets)]
    require(len(declarations) == 1, 'One literal replacement declaration is required in the actual diagnostic source')
    replacements = ast.literal_eval(declarations[0].value)
    require(len(replacements) == 2 and all(len(pair) == 2 and all(isinstance(v, str) for v in pair)
                                         for pair in replacements), 'Expected exactly two literal wording replacements')
    return replacements


def separation(rows, mode, variant):
    positive = [row[mode][variant]['p1'] for row in rows if row['observed_return'] == 1]
    negative = [row[mode][variant]['p1'] for row in rows if row['observed_return'] == 0]
    pm = statistics.mean(positive) if positive else None
    nm = statistics.mean(negative) if negative else None
    return dict(success_count=len(positive), failure_count=len(negative), success_mean_p1=pm,
                failure_mean_p1=nm, success_minus_failure_mean_p1=pm - nm if pm is not None and nm is not None else None)


def summarize_pool(rows):
    outcomes = [row['observed_return'] for row in rows]
    modes = {}
    for mode in ('original', 'corrected'):
        variants = {}
        for variant in ('factual', 'full_eos'):
            p1 = [row[mode][variant]['p1'] for row in rows]
            p0 = [row[mode][variant]['p0'] for row in rows]
            variants[variant] = dict(p0=describe(p0), p1=describe(p1),
                brier_p1=statistics.mean((p - g) ** 2 for p, g in zip(p1, outcomes)) if rows else None,
                brier_p0_vs_1_minus_G=statistics.mean((p - (1-g)) ** 2 for p, g in zip(p0, outcomes)) if rows else None,
                pearson_actual_G=pearson(p1, outcomes), success_failure_separation=separation(rows, mode, variant),
                pearson_p0_actual_G=pearson(p0, outcomes),
                two_class_probability_sum_minus_one=describe([row[mode][variant]['p0'] + row[mode][variant]['p1'] - 1 for row in rows]))
        modes[mode] = dict(variants=variants,
            factual_minus_full_eos_p1=describe([row[mode]['factual']['p1'] - row[mode]['full_eos']['p1'] for row in rows]),
            whole_response_success_label_log_ratio=describe([row[mode]['success_log_ratio'] for row in rows]),
            whole_response_observed_target_log_ratio=describe([row[mode]['observed_target_log_ratio'] for row in rows]))
    flips = [row for row in rows if row['original']['success_log_ratio'] * row['corrected']['success_log_ratio'] < 0]
    return dict(n=len(rows), actual_success_fraction=statistics.mean(outcomes) if rows else None, modes=modes,
        paired_corrected_minus_original=dict(
            factual_p0=describe([row['corrected']['factual']['p0'] - row['original']['factual']['p0'] for row in rows]),
            factual_p1=describe([row['corrected']['factual']['p1'] - row['original']['factual']['p1'] for row in rows]),
            full_eos_p0=describe([row['corrected']['full_eos']['p0'] - row['original']['full_eos']['p0'] for row in rows]),
            full_eos_p1=describe([row['corrected']['full_eos']['p1'] - row['original']['full_eos']['p1'] for row in rows]),
            success_label_log_ratio=describe([row['corrected']['success_log_ratio'] - row['original']['success_log_ratio'] for row in rows])),
        strict_nonzero_success_root_sign_flips=dict(count=len(flips), denominator=len(rows),
            identities=[dict(traj_uid=row['traj_uid'], source_step=row['source_step'], observed_return=row['observed_return']) for row in flips]),
        exact_zero_success_roots={mode: sum(row[mode]['success_log_ratio'] == 0 for row in rows) for mode in modes})


def previous_comparison(rows, previous, input_hash, sources):
    old_by_key = {}
    actual_imports = []
    checkpoints = []
    for rank in range(2):
        path = previous / f'rank{rank}-readout.json'
        run = read(path)
        sources.append(source(path))
        require(run['rank'] == rank and run['phase'] == 'complete_native_readout', 'Previous label readout must be complete')
        require(run['input_sha256'] == input_hash and run['outcome_token_ids'] == [15, 16], 'Previous original encoding/input differs')
        actual_imports.append(run['sources'])
        checkpoints.append(run['checkpoint'])
        for old in run['cases']:
            key = coordinate(old)
            require(key not in old_by_key, 'Duplicate previous UID/source step')
            old_by_key[key] = old
    require(len(old_by_key) == 64, 'Previous label readout coverage must be the same64')
    pairs = []
    for row in rows:
        old = old_by_key[coordinate(row)]
        require(old['observed_return'] == row['observed_return'] and old['source_start'] == row['source_start'] and
                old['source_end'] == row['source_end'] and old['context_tokens'] == row['original_context_tokens'] and
                ids_sha(old['original_query_ids']) == row['original_query_ids_sha256'], 'Previous original endpoint identity differs')
        item = dict(traj_uid=row['traj_uid'], source_step=row['source_step'])
        for variant, offset in (('factual', 0), ('full_eos', 1)):
            old_lp = old['native_outcome_log_probs'][offset]
            item[variant + '_lp_difference_by_class'] = [row['original'][variant]['log_probs'][i] - old_lp[i] for i in range(2)]
            item[variant + '_p_difference_by_class'] = [row['original'][variant][f'p{i}'] - math.exp(old_lp[i]) for i in range(2)]
        item['success_log_ratio_difference'] = item['factual_lp_difference_by_class'][1] - item['full_eos_lp_difference_by_class'][1]
        pairs.append(item)
    return dict(n=len(pairs), actual_imports=actual_imports, checkpoints=checkpoints,
        scope='Current original B2 pair versus the unexchanged original pair in prior label-encoding B4. Same saved IDs and physical classes; different native layouts. Descriptive differences, no tolerance verdict.',
        difference={variant: {kind: {str(i): describe([row[f'{variant}_{kind}_difference_by_class'][i] for row in pairs])
            for i in range(2)} for kind in ('lp', 'p')} for variant in ('factual', 'full_eos')},
        success_log_ratio_difference=describe([row['success_log_ratio_difference'] for row in pairs]), pairs=pairs)


def build_analysis(local, original_input, boundary_path, previous, driver_source):
    inputs, boundary = read(original_input), read(boundary_path)
    replacements = replacements_from_source(driver_source)
    input_hash, driver_hash = source(original_input)['sha256'], source(driver_source)['sha256']
    paths = [local / f'rank{rank}-readout.json' for rank in range(2)]
    completed_path = local / 'completed.json'
    runs, completed = [read(path) for path in paths], read(completed_path)
    sources = [source(path) for path in paths] + [source(completed_path), source(original_input),
        source(boundary_path), source(driver_source), source(Path(__file__))]
    for path in (driver_source.with_name('stage_textcraft_response_clock.py'),
                 AUDIT / 'verify_textcraft_native_readout.py',
                 ROOT / 'readout-quality-20261006/v2/readout-context-label-source-audit.json',
                 local / 'native-owner-inspection.json', local / 'prepared.json', local / 'job.json'):
        if path.is_file():
            sources.append(source(path))
    require(inputs['cases'] == boundary['rows'] == completed['cases'] == 64, 'Same64 coverage is required')
    require([len(rows) for rows in inputs['rank_cases']] == [32, 32], 'Original DataProto partition must remain32/rank')
    require(len(boundary['traj_uid']) == len(boundary['uid']) == 64, 'Original boundary group join must cover64')
    groups_by_uid = dict(zip(boundary['traj_uid'], boundary['uid']))
    require(len(groups_by_uid) == 64 and len(set(groups_by_uid.values())) == boundary['groups'] == 8, 'Expected original8 groups')
    require(all(completed[name] == 0 for name in ZERO_OPERATIONS) and completed['native_forward_calls'] == 128,
            'Completion must report128 native readouts and zero DT/backward/update')
    completed_ranks = {entry['rank']: entry for entry in completed['ranks']}
    require(set(completed_ranks) == {0, 1}, 'Completion needs both actual ranks')
    rows, original_queries, corrected_queries, original_texts, corrected_texts = [], set(), set(), set(), set()
    for rank, run in enumerate(runs):
        require(run['rank'] == rank and run['phase'] == 'complete_native_readout' and len(run['cases']) == 32,
                'Rank result must be complete with32 cases')
        require(run['input_sha256'] == input_hash and run['sampling'] == inputs['sampling'] and run['max_steps'] == inputs['max_steps'] and
                run['selection'] == inputs['selection'], 'Original input, sampling, horizon and selection must be retained')
        require(run['outcome_token_ids'] == [15, 16] and all(run[name] == 0 for name in ZERO_OPERATIONS), 'Classes/zero operations changed')
        require(run['sources']['observer']['sha256'] == run['sources']['query_text_adapter']['sha256'] == driver_hash,
                'Actual diagnostic source differs from the supplied source')
        require(run['replacements'] == [list(pair) for pair in replacements], 'Recorded replacements differ from actual diagnostic source')
        require(run['native_forward_calls'] == len(run['calls']) == 64, 'Each rank must make64 B2 calls')
        expected_order = [([index], mode) for index in range(32) for mode in ('original', 'corrected')]
        require([(call['case_indices'], call['mode']) for call in run['calls']] == expected_order, 'Native call order changed')
        require(completed_ranks[rank]['sha256'] == source(paths[rank])['sha256'] and
                completed_ranks[rank]['cases'] == 32 and completed_ranks[rank]['calls'] == 64, 'Completed rank hash/count mismatch')
        require([case['case_index'] for case in run['cases']] == list(range(32)), 'Original case order must be retained')
        for case in run['cases']:
            index = case['case_index']
            saved, query = inputs['rank_cases'][rank][index], case['query']
            require(coordinate(case) == coordinate(saved) and saved['source_step'] == 0, 'UID/first-response partition changed')
            require(boundary['traj_uid'][saved['actor_row']] == case['traj_uid'], 'Saved actor-row identity differs from original boundary')
            for field in ('observed_return', 'source_start', 'source_end'):
                require(case[field] == saved[field], 'Saved source/return changed: ' + field)
            require(saved['actual_G'] == case['observed_return'] in (0, 1), 'Original binary official return is required')
            original_ids, corrected_ids = query['original_ids'], query['corrected_ids']
            ids, start, end = saved['selected_input_ids'], saved['source_start'], saved['source_end']
            require(0 <= start < end < len(ids) and original_ids == ids[end:-1], 'Original query IDs differ from exact saved input')
            require(saved['outcome_token_ids'] == [15, 16] and query['observed_class_index'] == saved['observed_class_index'] == int(case['observed_return']) and
                    ids[-1] == saved['target_id'] == [15, 16][query['observed_class_index']], 'Original target/label meaning changed')
            expected_text = query['original_text']
            for old, new in replacements:
                require(expected_text.count(old) == 1 and new not in expected_text, 'Replacement is not a unique literal edit')
                expected_text = expected_text.replace(old, new)
            require(expected_text == query['corrected_text'], 'Candidate contains edits other than the two permitted clauses')
            restored = query['corrected_text']
            for old, new in replacements:
                require(restored.count(new) == 1, 'Replacement cannot be uniquely reversed')
                restored = restored.replace(new, old)
            require(restored == query['original_text'], 'Text changes are not exactly reversible')
            require(case['original_context_tokens'] == len(ids) and case['corrected_context_tokens'] == end + len(corrected_ids) + 1 <= 32768,
                    'Original/candidate predictor context mismatch')
            original_queries.add(tuple(original_ids)); corrected_queries.add(tuple(corrected_ids))
            original_texts.add(query['original_text']); corrected_texts.add(query['corrected_text'])
            record = dict(rank=rank, case_index=index, traj_uid=case['traj_uid'], source_step=case['source_step'],
                actor_row=saved['actor_row'], task_group_uid=groups_by_uid[case['traj_uid']], observed_return=case['observed_return'],
                source_start=start, source_end=end, retained_source_tokens=end-start,
                source_last_token_id=ids[end-1], source_len_equals_513=end-start == 513,
                original_source_ids_sha256=ids_sha(ids[start:end]), original_prefix_ids_sha256=ids_sha(ids[:end]),
                original_query_ids_sha256=ids_sha(original_ids), corrected_query_ids_sha256=ids_sha(corrected_ids),
                original_context_tokens=case['original_context_tokens'], corrected_context_tokens=case['corrected_context_tokens'])
            require(saved['retained_source_tokens'] == end-start, 'Saved retained source length mismatch')
            for mode, offset in (('original', 0), ('corrected', 1)):
                call = run['calls'][2*index + offset]
                require(call['shape'] == [2, case[mode + '_context_tokens'] - 1], 'Native pair has padding/wrong predictor shape')
                values = case['native_outcome_log_probs'][mode]
                require(len(values) == 2 and all(len(vector) == 2 for vector in values) and
                        all(math.isfinite(value) for vector in values for value in vector), 'Expected finite factual/EOS two-class log probabilities')
                fact, eos = values
                target_index = query['observed_class_index']
                record[mode] = dict(factual=dict(log_probs=fact, p0=math.exp(fact[0]), p1=math.exp(fact[1])),
                    full_eos=dict(log_probs=eos, p0=math.exp(eos[0]), p1=math.exp(eos[1])),
                    success_log_ratio=fact[1]-eos[1], observed_target_log_ratio=fact[target_index]-eos[target_index])
            rows.append(record)
    require(len({coordinate(row) for row in rows}) == 64 and {row['traj_uid'] for row in rows} == set(groups_by_uid), 'Duplicates or unmatched UIDs')
    require(len(original_queries) == len(corrected_queries) == len(original_texts) == len(corrected_texts) == 1,
            'All64 current step0 queries must be identical within each mode')
    require(runs[0]['checkpoint'] == runs[1]['checkpoint'], 'Ranks must use the same checkpoint')
    for key in set(runs[0]['sources']) & set(runs[1]['sources']):
        require(runs[0]['sources'][key] == runs[1]['sources'][key], 'Actual imported source differs across ranks: ' + key)
    groups = []
    for uid in sorted(set(groups_by_uid.values())):
        members = [row for row in rows if row['task_group_uid'] == uid]
        require(len(members) == 8, 'Original group is not n8')
        groups.append(dict(task_group_uid=uid, **summarize_pool(members)))
    within = {}
    for mode in ('original', 'corrected'):
        within[mode] = {}
        for variant in ('factual', 'full_eos'):
            residual_p, residual_g, mean_p, mean_g, gaps = [], [], [], [], []
            for group in groups:
                members = [row for row in rows if row['task_group_uid'] == group['task_group_uid']]
                p = [row[mode][variant]['p1'] for row in members]; g = [row['observed_return'] for row in members]
                pm, gm = statistics.mean(p), statistics.mean(g)
                residual_p.extend(value-pm for value in p); residual_g.extend(value-gm for value in g)
                mean_p.append(pm); mean_g.append(gm)
                gap = separation(members, mode, variant)['success_minus_failure_mean_p1']
                if gap is not None:
                    gaps.append(gap)
            within[mode][variant] = dict(within_group_pearson_actual_G=pearson(residual_p, residual_g),
                group_mean_forecast_vs_success_fraction_pearson=pearson(mean_p, mean_g),
                mixed_outcome_groups=len(gaps), successful_mean_below_failed_mean_groups=sum(gap < 0 for gap in gaps),
                within_group_success_minus_failure_mean_p1=describe(gaps))
    previous_summary = previous_comparison(rows, previous, input_hash, sources)
    require(all(checkpoint == runs[0]['checkpoint'] for checkpoint in previous_summary['checkpoints']), 'Previous checkpoint differs')
    for rank in range(2):
        for key in ('original_worker', 'producer', 'query_encoder', 'runner', 'native_reader', 'native_forward', 'native_precision'):
            if key in runs[rank]['sources'] or key in previous_summary['actual_imports'][rank]:
                require(runs[rank]['sources'].get(key) == previous_summary['actual_imports'][rank].get(key),
                        'Original owner changed across current/prior readouts: ' + key)
    source_audit_path = ROOT / 'readout-quality-20261006/v2/readout-context-label-source-audit.json'
    frozen_owners = read(source_audit_path)['frozen_owners'] if source_audit_path.is_file() else None
    prepared = read(local / 'prepared.json') if (local / 'prepared.json').is_file() else {}
    return dict(scope=__doc__, status='paired_response_clock_wording_observation_only', sources=sources,
        checkpoint=runs[0]['checkpoint'], selection=inputs['selection'], sampling=inputs['sampling'], max_steps=inputs['max_steps'],
        original_input_owners=inputs['inputs'], original_partition_owner=inputs['partition_owner'],
        original_boundary_owners=dict(snapshot=boundary['snapshot'], grpo_owner=boundary['grpo_owner']),
        actual_imports_by_rank=[run['sources'] for run in runs],
        frozen_owner_source_provenance=frozen_owners,
        prepared_source_checks=prepared.get('sources'), prepared_config_source=prepared.get('config_source'),
        runtime_by_rank=[dict(rank=run['rank'], pid=run['pid'], pid_birth=run['pid_birth'], model_dtype=run['model_dtype'],
            native_fla_fp16=run['native_fla_fp16'], attention=run['event_attention_backend'],
            categorical_log_prob_dtypes=sorted({call['categorical_log_prob_dtype'] for call in run['calls']}),
            allocation_before=run['allocation_before'], allocation_after=run['allocation_after']) for run in runs],
        calls=dict(native_forward_per_rank=[64,64], native_forward_total=128, native_reader_batch=2,
            finite_attributions=0, backward=0, optimizer=0, scheduler=0),
        source_contract=dict(original_queries_exact_and_identical=True, two_literal_edits_only=True, replacements_reversible=True,
            source_target_labels_unchanged=True, physical_outcome_labels=[15,16], success_class_index=1,
            original_response_and_prefix_arrays='Saved input SHA, source coordinates and executed diagnostic source contract; no separate model-input tensor capture is recorded.',
            actor_ppo_micro_batch_size_per_gpu=4, actor_B4_basis='The SHA-bound completed observer asserts the native actor config is4 before any readout.',
            DT_scope='No DT is invoked or configured by this diagnostic; formal DT B4 remains outside the B2 reader calls.',
            original_query_tokens=len(next(iter(original_queries))), corrected_query_tokens=len(next(iter(corrected_queries))),
            query_text=dict(original=next(iter(original_texts)), corrected=next(iter(corrected_texts)))),
        cases=64, observed_successes=sum(row['observed_return']==1 for row in rows), observed_failures=sum(row['observed_return']==0 for row in rows),
        task_groups=8, pools=dict(all=summarize_pool(rows), observed_success=summarize_pool([row for row in rows if row['observed_return']==1]),
            observed_failure=summarize_pool([row for row in rows if row['observed_return']==0])),
        task_group_pools=groups, within_group_discrimination=within,
        source_length_pools={name:{pool:summarize_pool([row for row in rows if predicate(row['retained_source_tokens']) and
            (outcome is None or row['observed_return'] == outcome)]) for pool,outcome in
            [('all',None),('observed_success',1),('observed_failure',0)]} for name,predicate in
            [('equals_513',lambda n:n==513), ('less_than_513',lambda n:n<513), ('greater_than_513',lambda n:n>513)]},
        source_length_scope='Retained original source length includes the owner-added im_end. 513 is the requested512-content-token length category, not proof of an unsaved engine stop reason.',
        previous_label_encoding_unexchanged_original=previous_summary, cases_by_uid_source_step=rows,
        completed_unix=completed['completed_unix'], limitations=[
            'One checkpoint and the saved64 first responses from eight dependent n8 task groups.',
            'Reported native head values are categorical log probabilities, not pre-softmax logits.',
            'Query-only original/corrected pairs use B2; the original actor B4 loss and DT token estimator are not evaluated here.',
            'Factual/EOS forecasts are approximate model readouts, not observed environment counterfactual probabilities.',
            'Larger endpoint differences or better in-batch Brier do not establish token credit accuracy or improved task PG.',
            'No per-token d/A, full-trajectory scatter, gradient, Adam update or historical causal result follows from these whole-response statistics.',
            'Success root sign flips are whole-response endpoint changes, not a rule requiring all successful actions to have positive credit.',
            'Comparison to prior B4 is descriptive and is not an official FA/FLA tolerance test or a repair gate.',
            'Group-centering is used only for descriptive within-group correlation; no advantage or probability is rescaled.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, default=LOCAL)
    parser.add_argument('--original-input', type=Path, default=INPUT)
    parser.add_argument('--boundary', type=Path, default=BOUNDARY)
    parser.add_argument('--previous-dir', type=Path, default=PREVIOUS)
    parser.add_argument('--driver-source', type=Path, default=DRIVER)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_analysis(args.local_dir, args.original_input, args.boundary, args.previous_dir, args.driver_source)
    output = args.output or args.local_dir / 'response-clock-analysis.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    clock_summary = {mode: dict(
        factual_mean_p1=result['pools']['all']['modes'][mode]['variants']['factual']['p1']['mean'],
        factual_brier=result['pools']['all']['modes'][mode]['variants']['factual']['brier_p1'],
        success_root_abs_mean=result['pools']['observed_success']['modes'][mode]['whole_response_success_label_log_ratio']['abs_mean'])
        for mode in ('original', 'corrected')}
    prior_lp_summary = {variant: {label: dict(mean=stats['mean'], mean_abs=stats['abs_mean'],
        max_abs=max(abs(stats['min']), abs(stats['max']))) for label, stats in
        result['previous_label_encoding_unexchanged_original']['difference'][variant]['lp'].items()}
        for variant in ('factual', 'full_eos')}
    print(json.dumps(dict(path=str(output.resolve()), sha256=source(output)['sha256'], cases=result['cases'],
        successes=result['observed_successes'], failures=result['observed_failures'], clock_readout=clock_summary,
        prior_B2_minus_B4_log_probability_difference=prior_lp_summary,
        scope='Whole-response LP observation only; B2/B4 differences are descriptive; no token-credit or PG result.'),
        ensure_ascii=False, allow_nan=False))
