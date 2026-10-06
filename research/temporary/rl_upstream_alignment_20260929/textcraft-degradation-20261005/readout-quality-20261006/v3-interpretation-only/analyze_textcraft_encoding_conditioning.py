"""Read saved paired class encodings to describe prefix/reference interactions.

This is a standard-library analysis of native read_outcomes log probabilities.
It does not run a model, change readout or credit, fit a replacement predictor,
or treat its algebraic contrasts as causal attributions.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from analyze_textcraft_native_group_label_response import ids_hash, separation, source

AUDIT = Path(__file__).resolve().parent
DEG = AUDIT / 'textcraft-degradation-20261005'
NATIVE = DEG / 'readout-quality-20261006/v2'
LABEL = DEG / 'label-encoding-20261006/v2'


def describe(values):
    values = list(values)
    mean = math.fsum(values) / len(values)
    return dict(count=len(values), mean=mean, mean_absolute=math.fsum(map(abs, values)) / len(values),
        population_standard_deviation=statistics.pstdev(values),
        root_mean_square=math.sqrt(math.fsum(x*x for x in values) / len(values)),
        minimum=min(values), maximum=max(values), positive=sum(x > 0 for x in values),
        negative=sum(x < 0 for x in values), zero=sum(x == 0 for x in values))


def correlation(a, b):
    ma, mb = math.fsum(a)/len(a), math.fsum(b)/len(b)
    aa, bb = [x-ma for x in a], [x-mb for x in b]
    denominator = math.sqrt(math.fsum(x*x for x in aa) * math.fsum(x*x for x in bb))
    return math.fsum(x*y for x,y in zip(aa,bb))/denominator if denominator else None


def summarize(rows):
    fields = ('old_factual_success_probability', 'old_EOS_success_probability',
              'new_factual_success_probability', 'new_EOS_success_probability',
              'factual_success_probability_encoding_shift', 'EOS_success_probability_encoding_shift',
              'old_factual_success_log_odds', 'old_EOS_success_log_odds',
              'new_factual_success_log_odds', 'new_EOS_success_log_odds',
              'factual_success_log_odds_encoding_shift', 'EOS_success_log_odds_encoding_shift',
              'old_factual_minus_EOS_log_odds', 'new_factual_minus_EOS_log_odds',
              'factual_minus_EOS_log_odds_encoding_interaction',
              'old_success_logp_contrast', 'new_success_logp_contrast',
              'old_observed_category_logp_contrast', 'new_observed_category_logp_contrast')
    return dict(rows=len(rows), observed_successes=sum(r['G'] for r in rows),
        descriptions={key:describe(r[key] for r in rows) for key in fields},
        same_case_factual_vs_EOS_encoding_shift_correlation=dict(
            probability=correlation([r['factual_success_probability_encoding_shift'] for r in rows],
                                    [r['EOS_success_probability_encoding_shift'] for r in rows]),
            log_odds=correlation([r['factual_success_log_odds_encoding_shift'] for r in rows],
                                 [r['EOS_success_log_odds_encoding_shift'] for r in rows])),
        success_event_logp_contrast_sign_flips=sum(r['old_success_logp_contrast'] * r['new_success_logp_contrast'] < 0 for r in rows),
        observed_category_logp_contrast_sign_flips=sum(r['old_observed_category_logp_contrast'] * r['new_observed_category_logp_contrast'] < 0 for r in rows),
        realized_return_separation={key:separation(rows,key) for key in fields[:4]})


def build_analysis():
    paths = dict(cases=NATIVE/'original-first-response-cases.json',
        boundary=DEG/'native-minibatch-v4/native-minibatch-update-boundary.json',
        label_inspection=LABEL/'native-owner-inspection.json',
        label_completed=LABEL/'completed.json',
        existing_group_review=NATIVE/'native64-task-group-label-response.json',
        **{f'old_rank{rank}':NATIVE/f'rank{rank}-readout.json' for rank in (0,1)},
        **{f'label_rank{rank}':LABEL/f'rank{rank}-readout.json' for rank in (0,1)})
    data = {name:json.loads(path.read_bytes()) for name,path in paths.items()}
    inputs = [c for cases in data['cases']['rank_cases'] for c in cases]
    grouped_uid = dict(zip(data['boundary']['traj_uid'], data['boundary']['uid']))
    old = {(c['traj_uid'],c['source_step']):c for rank in (0,1) for c in data[f'old_rank{rank}']['cases']}
    paired = {(c['traj_uid'],c['source_step']):c for rank in (0,1) for c in data[f'label_rank{rank}']['cases']}
    existing = {r['traj_uid']:r for r in data['existing_group_review']['rows']}
    assert len(inputs) == len(old) == len(paired) == len(grouped_uid) == 64
    assert data['label_inspection']['only_two_label_ids_changed']
    rows, groups = [], {}
    for case in inputs:
        uid, step = case['traj_uid'], case['source_step']
        nc, lc = old[(uid,step)], paired[(uid,step)]
        assert step == nc['source_step'] == lc['source_step'] == 0
        assert case['actual_G'] == nc['observed_return'] == lc['observed_return']
        assert lc['success_class_indices'] == [1,1,0,0]
        start, end, ids = case['source_start'], case['source_end'], case['selected_input_ids']
        assert start == nc['source_start'] == lc['source_start'] and end == nc['source_end'] == lc['source_end']
        assert ids[end:-1] == lc['original_query_ids']
        assert len(lc['original_query_ids']) == len(lc['swapped_query_ids'])
        changed = [i for i,(a,b) in enumerate(zip(lc['original_query_ids'],lc['swapped_query_ids'])) if a!=b]
        assert sorted((lc['original_query_ids'][i],lc['swapped_query_ids'][i]) for i in changed) == [(15,16),(16,15)]
        lp = lc['native_outcome_log_probs']
        semantic_lp = [lp[i][col] for i,col in enumerate(lc['success_class_indices'])]
        semantic_log_odds = [lp[i][col]-lp[i][1-col] for i,col in enumerate(lc['success_class_indices'])]
        probabilities = list(map(math.exp,semantic_lp))
        observed_lp = lc['native_target_log_probs']
        row = dict(traj_uid=uid, prompt_uid=grouped_uid[uid], source_step=step,
            prompt_ids_sha256=ids_hash(ids[:start]), source_tokens=end-start, G=case['actual_G'],
            official_item_id=existing[uid]['official_item_id'],
            original_paired_native_input_shape=lc['native_input_shape'],
            original_query_tokens=len(lc['original_query_ids']), changed_query_positions=changed,
            categorical_log_probs=lp, semantic_success_log_probs=semantic_lp,
            **dict(zip(('old_factual_success_probability','old_EOS_success_probability',
                        'new_factual_success_probability','new_EOS_success_probability'),probabilities)),
            **dict(zip(('old_factual_success_log_odds','old_EOS_success_log_odds',
                        'new_factual_success_log_odds','new_EOS_success_log_odds'),semantic_log_odds)),
            factual_success_probability_encoding_shift=probabilities[2]-probabilities[0],
            EOS_success_probability_encoding_shift=probabilities[3]-probabilities[1],
            factual_success_log_odds_encoding_shift=semantic_log_odds[2]-semantic_log_odds[0],
            EOS_success_log_odds_encoding_shift=semantic_log_odds[3]-semantic_log_odds[1],
            old_factual_minus_EOS_log_odds=semantic_log_odds[0]-semantic_log_odds[1],
            new_factual_minus_EOS_log_odds=semantic_log_odds[2]-semantic_log_odds[3],
            factual_minus_EOS_log_odds_encoding_interaction=(semantic_log_odds[2]-semantic_log_odds[3])-(semantic_log_odds[0]-semantic_log_odds[1]),
            old_success_logp_contrast=semantic_lp[0]-semantic_lp[1],
            new_success_logp_contrast=semantic_lp[2]-semantic_lp[3],
            old_observed_category_logp_contrast=observed_lp[0]-observed_lp[1],
            new_observed_category_logp_contrast=observed_lp[2]-observed_lp[3],
            paired_original_vs_prior_native_factual_probability=probabilities[0]-math.exp(nc['native_outcome_log_probs'][0][1]),
            paired_original_vs_prior_native_EOS_probability=probabilities[1]-math.exp(nc['native_outcome_log_probs'][1][1]))
        assert row['prompt_ids_sha256'] == existing[uid]['prompt_ids_int64_le_sha256']
        rows.append(row)
        groups.setdefault(grouped_uid[uid],[]).append(row)
    assert len(groups) == 8
    group_reports = []
    for uid, cases in groups.items():
        assert len(cases) == 8 and len({c['prompt_ids_sha256'] for c in cases}) == 1
        group_reports.append(dict(prompt_uid=uid, prompt_ids_sha256=cases[0]['prompt_ids_sha256'],
            official_item_id=cases[0]['official_item_id'], **summarize(cases)))
    shifts = [row[key] for row in rows for key in ('factual_success_log_odds_encoding_shift','EOS_success_log_odds_encoding_shift')]
    centers = {uid:math.fsum(c[key] for c in cases for key in ('factual_success_log_odds_encoding_shift','EOS_success_log_odds_encoding_shift'))/(2*len(cases)) for uid,cases in groups.items()}
    centered = [row[key]-centers[row['prompt_uid']] for row in rows for key in ('factual_success_log_odds_encoding_shift','EOS_success_log_odds_encoding_shift')]
    separation_keys = tuple(group_reports[0]['realized_return_separation'])
    return dict(scope='Read-only native paired probability/encoding conditioning audit on one checkpoint and saved native64; no model, DT, backward or credit changes.',
        analysis_source=source(Path(__file__)), descriptive_helper_source=source(AUDIT/'analyze_textcraft_native_group_label_response.py'),
        source_files={name:source(path) for name,path in paths.items()},
        original_model_sources={name:data[name]['sources'] for name in ('old_rank0','old_rank1','label_rank0','label_rank1')},
        checkpoint=data['label_rank0']['checkpoint'], original_input_sha256=data['label_rank0']['input_sha256'],
        population=dict(rows=64, prompt_groups=8, per_group=8, successful21=sum(r['G'] for r in rows),
            failure43=sum(r['G']==0 for r in rows), source_step=0),
        probability_definition='exp(actual two-category FP32 log-prob); semantic success is original column1/swapped column0. Semantic log-odds are actual success LP minus failure LP, not separately saved vocabulary logits.',
        pooled=summarize(rows), by_actual_prompt=group_reports,
        descriptive_constant_log_odds_shift=dict(all_fact_and_EOS_128=describe(shifts),
            equal_prompt_mean_log_odds_shifts=centers,
            residuals_after_each_prompt_common_fact_and_EOS_shift=describe(centered),
            scope='Mean/dispersion describe whether one shared additive shift fits the observations; no shift is subtracted from any production probability or credit, and this is not causal attribution.'),
        within_prompt_separation={key:dict(
            success_failure_pairs=sum(g['realized_return_separation'][key]['success_failure_pairs'] for g in group_reports),
            pair_weighted_AUC=math.fsum(g['realized_return_separation'][key]['pairwise_wins_with_half_ties'] for g in group_reports)/sum(g['realized_return_separation'][key]['success_failure_pairs'] for g in group_reports),
            equal_prompt_success_minus_failure=math.fsum(g['realized_return_separation'][key]['success_minus_failure'] for g in group_reports)/len(group_reports)) for key in separation_keys},
        prior_native_original_encoding_comparison=dict(
            factual_probability=describe(r['paired_original_vs_prior_native_factual_probability'] for r in rows),
            EOS_probability=describe(r['paired_original_vs_prior_native_EOS_probability'] for r in rows)),
        rows=rows, operations=dict(model_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0),
        limitations=[
            'These are paired class-encoding/native readout observations, not token d, Q/V/A, policy-gradient components or overall attribution quality.',
            'The observed log-odds shifts are incompatible with a single fixed additive semantic-success/failure logit bias applied unchanged to every factual and EOS endpoint; probability-shift variation alone would not establish that because sigmoid is nonlinear.',
            'The permutation changes two literal query label tokens as well as the semantic target-column mapping. Query interpretation, its interaction with the factual/EOS prefix, or other input-dependent label preferences can produce the remaining shifts; the algebra does not isolate a source-token causal effect.',
            'The 29/64 sign-flip count is for the whole-response factual-minus-EOS success-event log-prob contrast under two queries, not a per-token error rate.',
            'A nonconstant encoding shift or factual/reference interaction does not identify its causal internal mechanism.',
            'EOS references preserve each response length; length and encoded forecast context can contribute to the descriptive variation.',
            'One saved rollout with eight tasks and one realized future per trajectory cannot establish absent world-state conditioning or future calibration.',
            'No encoding is selected, averaged, normalized or deployed, and no numerical tolerance is defined.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=NATIVE/'encoding-conditioning-audit.json')
    args = parser.parse_args()
    result = build_analysis()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(args.output),sha256=source(args.output)['sha256'],
        rows=result['population'], descriptions=result['pooled']['descriptions'],
        within_prompt_separation=result['within_prompt_separation'])))
