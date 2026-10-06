"""Describe saved first-response EOS reference dependence; no model or gate."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


LOCAL = Path(__file__).resolve().parent
DEG = LOCAL.parents[1]


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def summary(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None,
                range=max(values)-min(values) if values else None,
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values))


def pearson(x, y):
    if len(x) < 2:
        return None
    xc, yc = [v-statistics.mean(x) for v in x], [v-statistics.mean(y) for v in y]
    xx, yy = math.fsum(v*v for v in xc), math.fsum(v*v for v in yc)
    return math.fsum(a*b for a, b in zip(xc, yc))/math.sqrt(xx*yy) if xx and yy else None


METRICS = ('reference_success_logp', 'reference_success_probability',
           'factual_success_probability', 'success_log_ratio',
           'success_probability_difference', 'observed_target_log_ratio')


def describe(rows):
    lengths = [r['source_tokens'] for r in rows]
    return dict(cases=len(rows), prompt_groups=len({r['prompt_sha256'] for r in rows}),
                actual_success_fraction=statistics.mean(r['actual_G'] for r in rows) if rows else None,
                source_tokens=summary(lengths),
                metrics={key: summary(r[key] for r in rows) for key in METRICS},
                pearson_length={key: pearson(lengths, [r[key] for r in rows]) for key in METRICS},
                pearson_outcome={key: pearson([r['actual_G'] for r in rows], [r[key] for r in rows])
                                 for key in ('source_tokens',)+METRICS})


def main():
    started = time.time()
    input_path = LOCAL/'original-first-response-cases.json'
    boundary_path = DEG/'native-minibatch-v4/native-minibatch-update-boundary.json'
    raw_paths = [DEG/f'label-encoding-20261006/v2/rank{rank}-readout.json' for rank in range(2)]
    pack = json.loads(input_path.read_bytes())
    boundary = json.loads(boundary_path.read_bytes())
    cases = {(r['traj_uid'], r['source_step']): r for rank in pack['rank_cases'] for r in rank}
    group_by_uid = dict(zip(boundary['traj_uid'], boundary['uid']))
    assert len(cases) == len(group_by_uid) == 64
    runs = [json.loads(p.read_bytes()) for p in raw_paths]
    rows, exact_prompts, exact_queries = [], {}, set()
    groups, same_inputs = defaultdict(list), defaultdict(list)
    for rank, run in enumerate(runs):
        assert run['phase'] == 'complete_native_readout' and len(run['cases']) == 32
        assert run['input_sha256'] == identity(input_path)['sha256']
        assert run['optimizer_steps'] == run['scheduler_steps'] == run['backward_calls'] == run['finite_trace_calls'] == 0
        assert run['outcome_token_ids'] == [15, 16] and run['native_forward_calls'] == 32
        for raw in run['cases']:
            key = raw['traj_uid'], raw['source_step']
            saved = cases[key]
            ids, start, end = saved['selected_input_ids'], saved['source_start'], saved['source_end']
            assert key[1] == 0 and raw['source_start'] == start and raw['source_end'] == end
            assert raw['observed_return'] == saved['observed_return'] and raw['success_class_indices'][:2] == [1, 1]
            prompt, query = tuple(ids[:start]), tuple(ids[end:-1])
            assert tuple(raw['original_query_ids']) == query
            assert ids[-1] == [15, 16][saved['observed_class_index']]
            assert ids[end-1] == 248046
            native_group = group_by_uid[key[0]]
            assert native_group not in exact_prompts or exact_prompts[native_group] == prompt
            exact_prompts[native_group] = prompt
            exact_queries.add(query)
            prompt_hash = ids_sha(prompt)
            reference = list(prompt)+[248046]*(end-start)+list(query)
            fact_lp, ref_lp = raw['native_outcome_log_probs'][:2]
            assert len(fact_lp) == len(ref_lp) == 2
            observed = saved['observed_class_index']
            row = dict(rank=rank, case_index=raw['case_index'], actor_row=saved['actor_row'],
                traj_uid=key[0], source_step=0, native_group=native_group,
                actual_G=saved['observed_return'], source_tokens=end-start,
                prompt_tokens=start, query_tokens=len(query), prompt_sha256=prompt_hash,
                literal_reference_input_sha256=ids_sha(reference),
                reference_success_logp=ref_lp[1], reference_success_probability=math.exp(ref_lp[1]),
                factual_success_probability=math.exp(fact_lp[1]), success_log_ratio=fact_lp[1]-ref_lp[1],
                success_probability_difference=math.exp(fact_lp[1])-math.exp(ref_lp[1]),
                observed_target_log_ratio=fact_lp[observed]-ref_lp[observed],
                native_input_shape=raw['native_input_shape'])
            rows.append(row)
            groups[prompt_hash].append(row)
            same_inputs[(prompt_hash, end-start)].append(row)
    assert len(rows) == 64 and len({r['traj_uid'] for r in rows}) == 64
    assert len(exact_prompts) == len(groups) == 8 and len(exact_queries) == 1
    assert all(len(group) == 8 for group in groups.values())
    repeated = []
    for (prompt_hash, length), members in same_inputs.items():
        if len(members) < 2:
            continue
        assert len({r['literal_reference_input_sha256'] for r in members}) == 1
        repeated.append(dict(prompt_sha256=prompt_hash, source_tokens=length,
            literal_reference_input_sha256=members[0]['literal_reference_input_sha256'],
            cases=len(members), ranks=sorted({r['rank'] for r in members}),
            reference_success_logp=summary(r['reference_success_logp'] for r in members),
            reference_success_probability=summary(r['reference_success_probability'] for r in members),
            members=[dict(traj_uid=r['traj_uid'], rank=r['rank'], actual_G=r['actual_G']) for r in members]))
    centered = {key: [] for key in ('source_tokens', 'actual_G')+METRICS}
    for members in groups.values():
        for field in centered:
            mean = statistics.mean(r[field] for r in members)
            centered[field].extend(r[field]-mean for r in members)
    result = dict(scope='Saved native label B4 original mapping only. Entire-response factual/EOS endpoints; no single-token result, model, GPU, training or tolerance gate.',
        sources=dict(script=identity(Path(__file__)), literal_inputs=identity(input_path),
            native_uid_groups=identity(boundary_path), raw=[identity(p) for p in raw_paths]),
        owner_sources=[r['sources'] for r in runs],
        model_scope={k:runs[0][k] for k in ('checkpoint', 'sampling', 'max_steps', 'outcome_token_ids',
            'model_dtype', 'native_fla_fp16', 'event_attention_backend', 'numerical_runner_options')},
        geometry=dict(cases=64, unique_UIDs=64, exact_native_prompt_groups=8,
            exact_distinct_prompt_ID_sequences=len(set(exact_prompts.values())),
            rows_per_prompt=8, exact_distinct_query_ID_sequences=len(exact_queries),
            EOS_token_id=248046, response_length_includes_original_appended_EOS=True,
            fixed_513='512 native output tokens plus original handler EOS; not a new cutoff'),
        all=describe(rows), fixed_length_513=describe([r for r in rows if r['source_tokens']==513]),
        shorter_than_513=describe([r for r in rows if r['source_tokens']<513]),
        by_actual_G={str(g):describe([r for r in rows if r['actual_G']==g]) for g in (0, 1)},
        by_prompt=[dict(prompt_sha256=k, native_group=v[0]['native_group'], description=describe(v),
            fixed_513=describe([r for r in v if r['source_tokens']==513]),
            shorter_than_513=describe([r for r in v if r['source_tokens']<513])) for k,v in groups.items()],
        within_prompt_centered_pearson=dict(length={key:pearson(centered['source_tokens'], centered[key]) for key in METRICS},
            outcome={key:pearson(centered['actual_G'], centered[key]) for key in ('source_tokens',)+METRICS}),
        same_prompt_same_length=dict(groups=len(repeated), cases=sum(r['cases'] for r in repeated),
            max_reference_LP_range=max((r['reference_success_logp']['range'] for r in repeated), default=None),
            max_reference_probability_range=max((r['reference_success_probability']['range'] for r in repeated), default=None), rows=repeated),
        rows=rows, limitations=[
            'Only the original legend rows from the already-run label B4 are used; swapped-label rows are excluded.',
            'Reference input equals exact prompt + current-response-length repeated EOS + identical forecast query. Different lengths are different model inputs.',
            'Pooled correlations mix tasks; within-prompt centering is descriptive, not randomization or a causal effect.',
            'Same prompt/length inputs may differ in batch partners, rank or invocation. Report raw dispersion, not an invented tolerance.',
            'Observed dependence does not prove EOS is necessarily biased, locate all weak task-gradient causes, or validate whole DT.',
            'Root here is a whole-response probability/log-probability contrast, never a single-token advantage.'],
        operations=dict(model_calls=0, GPU_calls=0, DT_calls=0, backward=0, optimizer=0, production_changes=0),
        elapsed_seconds=time.time()-started)
    output = LOCAL/'eos-reference-length-audit.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=identity(output), all=result['all'],
        within_prompt=result['within_prompt_centered_pearson'], same_input={k:v for k,v in result['same_prompt_same_length'].items() if k!='rows'})))


if __name__ == '__main__':
    main()
