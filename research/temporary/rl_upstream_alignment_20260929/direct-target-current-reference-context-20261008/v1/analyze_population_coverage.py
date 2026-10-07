"""Join existing real endpoint probes to saved-population coefficient statistics.

Only descriptive CPU arithmetic; no training credit, query policy or tolerance.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
AUDIT = HERE.parents[1]


def binding(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


population_path = AUDIT / 'direct-target-credit-sample-20261007/v1/population.json'
old_reference_path = AUDIT / 'direct-target-reference-interaction-20261007/v1/analysis.json'
current_path = REPO / 'experiments/rl/results_current_token_reference_context_20261008.json'
causal_path = REPO / 'experiments/rl/results_credit_causal_diagnosis_20261008.json'
text_curve_path = REPO / 'experiments/rl/results_textcraft_actual_author_curves_20261008.json'
gradient_path = REPO / 'experiments/rl/results_update_gradient_20261008.json'
population = json.loads(population_path.read_bytes())
old = json.loads(old_reference_path.read_bytes())
current = json.loads(current_path.read_bytes())
causal = json.loads(causal_path.read_bytes())
text_curve = json.loads(text_curve_path.read_bytes())
gradient = json.loads(gradient_path.read_bytes())

known = [
    dict(task='appworld', uid=current['case']['uid'],
         packed_slot=current['case']['packed_slot'], token_id=current['case']['token_id'],
         native_sha256=current['case']['native_sha256'],
         source_sha256=current['case']['source_sha256'],
         native_d=current['factual_context_effect'], receipt=binding(current_path)),
    dict(task='appworld', uid=old['candidate']['traj_uid'],
         packed_slot=old['candidate']['packed_slot'], token_id=old['candidate']['token_id'],
         native_sha256=old['provenance']['native_sha256'],
         source_sha256=old['provenance']['source_sha256'],
         native_d=old['factual_context']['d'], receipt=binding(old_reference_path)),
    dict(task='textcraft', uid=causal['selected_actual_tokens']['TextCraft_Format']['traj_uid'],
         packed_slot=causal['selected_actual_tokens']['TextCraft_Format']['packed_slot'],
         token_id=causal['selected_actual_tokens']['TextCraft_Format']['token_id'],
         native_sha256=text_curve['native_sha256'], source_sha256=text_curve['source_sha256'],
         native_d=causal['selected_actual_tokens']['TextCraft_Format']['native_single_delete_d'],
         receipt=binding(causal_path)),
]
matched = []
for case in known:
    task = population['tasks'][case['task']]
    assert task['source']['sha256'] == case['source_sha256']
    hits = []
    for batch in task['batches']:
        for row in batch['rows']:
            if row['traj_uid'] != case['uid']:
                continue
            point = row['candidates']['most_negative']
            assert batch['file']['sha256'] == case['native_sha256']
            assert point['packed_slot'] == case['packed_slot'] and point['token_id'] == case['token_id']
            future_targets = [j for j in row['target_offsets'] if j > point['response_slot']]
            hits.append(dict(**case, formal_saved_point=point,
                coefficient_square=point['expected_A_FP32']**2,
                next_target_distance=min(future_targets)-point['response_slot'] if future_targets else None,
                native_file=batch['file'], native_row=row['batch_row']))
    assert len(hits) == 1
    matched.extend(hits)

tasks = {}
for name, task in population['tasks'].items():
    rows = [row for batch in task['batches'] for row in batch['rows']]
    rows.sort(key=lambda row:row['negative_prior_sumsq'], reverse=True)
    points = [p for p in matched if p['task'] == name]
    square = sum(p['coefficient_square'] for p in points)
    prior_negative_square = task['prior_A']['negative_sumsq']
    tasks[name] = dict(
        saved_native_files=task['native_files'], saved_request_rows=task['saved_trajectory_rows'],
        distinct_UIDs=task['unique_traj_uids'], duplicate_request_rows=task['duplicate_traj_rows'],
        prior_source_slots=task['prior_A']['count'], all_policy_slots=task['policy_A']['count'],
        prior_negative_slots=task['prior_A']['negative_count'],
        matched_actual_native_probes=points,
        covered_negative_coefficient_square=square,
        covered_fraction_of_negative_prior_square=square/prior_negative_square,
        covered_fraction_of_all_prior_square=square/task['prior_A']['sumsq'],
        covered_fraction_of_all_policy_square=square/task['policy_A']['sumsq'],
        descriptive_original_A_le_minus5=task['prior_A']['descriptive_thresholds']['5'],
        matched_A_le_minus5_count=sum(p['formal_saved_point']['expected_A_FP32'] <= -5 for p in points),
        matched_opposite_sign_count=sum((p['formal_saved_point']['d_FP64'] < 0) != (p['native_d'] < 0) for p in points),
        row_concentration={str(n):sum(r['negative_prior_sumsq'] for r in rows[:n])/prior_negative_square
                           for n in (1,2,5,10)},
        scope=task['scope'])

result = dict(
    status='Existing measured endpoint probes joined to saved population; no new GPU run',
    tasks=tasks,
    actual_TextCraft_gradient=dict(scope=gradient['scope'], metrics=gradient['gradient'],
                                  source=binding(gradient_path)),
    conclusion=[
        'Both AppWorld saved source coefficients <= -5 are now linked to existing positive native factual-context deletion effects by exact UID, position, token, input SHA and frozen source SHA.',
        'The two AppWorld sources cover the bulk of negative-source coefficient squares in the completed saved subset, but a much smaller share of all-policy squares. Neither fraction is a parameter-gradient share.',
        'TextCraft Format has a measured negative native effect with overestimated magnitude. Its native measured gradient contribution is separately retained, rather than inferred from coefficient squares.',
        'These measurements narrow the extreme-credit problem to specific actual token effects; they do not establish the overall learning-degradation cause, a population sign-error rate or that all attribution is bad.'],
    limitations=[
        'AppWorld formal DT did not finish: population covers its completed 54 native captures only.',
        'TextCraft statistics weight 176 saved request rows, including six repeated UIDs; this is not the actor global batch or a deduplicated episode estimate.',
        'The -5 bin is inherited descriptive reporting, not a new query rule, clipping threshold or acceptance criterion.',
        'Native endpoint probes are model causal diagnostics under the chosen EOS intervention, not newly executed environment rollouts.',
        'Original cumulative deletion, RISE/MAS and official FA/FLA numerical assertions are not replaced by this diagnostic join.'],
    source_bindings=[binding(p) for p in [population_path,old_reference_path,current_path,causal_path,text_curve_path]],
    operations=dict(new_model_forward=0, DT=0, backward=0, optimizer=0,
                    new_rollout=0, credit_changed=False, formal_restart=False))
(HERE/'population-coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({name:{k:v for k,v in task.items() if k not in ('matched_actual_native_probes','scope')}
                  for name,task in tasks.items()},ensure_ascii=False))
