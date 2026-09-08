"""Diagnose the failed NI0 adaptation using frozen artifacts; zero GPU calls.

This does not select a new method, alter gold, or evaluate new deletion curves.
Layer and target decompositions are descriptive diagnostics, not new benchmarks.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
receipts = {}


def read(name, expected=None, numpy=False):
    path = A / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, name
    receipts[name] = digest
    return np.load(path) if numpy else json.loads(path.read_bytes())


old = read('snapshot${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1/results.json',
           '679842f0ccfc32769a416868f6c08f70ddffff84298b80b02f02ca9da1457476')
old = next(r for r in old['records'] if r['dataset'] == 'niah_mq_q2' and r['idx'] == 0)
spans = read('snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json',
             '37a0d8c4ef4d174ebabd346087be303e895a0eaa78ca9231f2a6ba3195890b6a')
quality = read('snapshot${ARTIFACT_ROOT}/codex_qwen35_saved_recovery_20260908_v1/results.json',
               'e38afa4031de06833a17732065f5e029081d8f569e604c5a1821d0834bc377a8')
root = read('snapshot${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v2/results.json',
            '55378a7d777fa500631614afcf019fcfb42e4363172e65e92c370305b49229fe')
tokens = read('snapshot/tmp/qwen35_shared_failure_tokens_20260908.json',
              '1fc4951539608a2d83932f0432e66964f093ee9a51e033d5cb651718aa471492')
vectors = read('snapshot${ARTIFACT_ROOT}/codex_qwen35_saved_recovery_20260908_v1/immutable_score_vectors.npz',
               quality['vector_sha256'], numpy=True)
dataset = A / 'snapshot${FLASHTRACE_ROOT}/exp/exp2/data/niah_mq_q2.jsonl'
receipts[str(dataset.relative_to(A))] = hashlib.sha256(dataset.read_bytes()).hexdigest()
assert receipts[str(dataset.relative_to(A))] == spans['protocol']['cache_sha256']['niah_mq_q2']
record = json.loads(dataset.read_text(encoding='utf-8').splitlines()[0])

result = {
    'status': 'both_current_adaptations_fail_NI0_evidence_recovery_acceptance',
    'scope': 'One identical author NI0 text. Cross-checkpoint historical values are descriptive, not a controlled architecture ablation.',
    'new_model_loads': 0, 'new_model_forwards': 0, 'new_attribution_calls': 0,
    'new_metric_calls': 0, 'tokenizer_only_attempts': 2,
    'tokenizer_attempt_note': 'First CPU tokenizer import failed before tokenization because MACA_PATH was unset; retry with existing required environment succeeded. No model/GPU computation.',
    'old_8B': {}, 'new_9B': {}, 'target_effects': [],
    'score_transform': 'Original author positive projection only; unchanged signed vectors retained.',
}

for token_row in tokens['cases']:
    assert token_row['prompt_text_sha256'] == hashlib.sha256(record['prompt'].encode()).hexdigest()
old_token, new_token = tokens['cases']
assert old_token['user_ids'] == [old['input_ids'][j] for j in old['user_positions']]
mapping = spans['cases'][0]['mapping']
assert mapping['old_gold_user_token_indices'] == old['gold_full_local']
assert len(old['keep_local_indices']) == len(mapping['keep_local_indices']) == 310
assert len(old['gold_eligible_local']) == 40


def describe(score, selected, keep, gold, token_row):
    score = np.asarray(score, dtype=np.float64)
    keep = np.asarray(keep, dtype=int)
    gold = set(gold)
    question = set(token_row['question_local_positions'])
    pos = np.maximum(score, 0)
    denominator = float(pos[keep].sum())
    g = sorted(gold)
    return {
        'recovery': len(set(selected) & gold) / len(gold),
        'gold_hits': len(set(selected) & gold), 'eligible_gold_count': len(gold),
        'selected_count': len(selected),
        'selected_in_final_question': len(set(selected) & question),
        'question_eligible_count': len(set(keep) & question),
        'positive_mass_fraction_in_question': float(pos[sorted(set(keep) & question)].sum()) / denominator,
        'positive_mass_fraction_in_gold': float(pos[g].sum()) / denominator,
        'signed_gold_sum': float(score[g].sum()),
        'negative_gold_count': int((score[g] < 0).sum()),
        'top_ten_token_text': [token_row['user_tokens'][i] for i in selected[:10]],
    }


for method, selected in old['recovery_topk_local'].items():
    if method in old['native']:
        score = np.asarray(old['native'][method]['signed_full_sequence'])[old['user_positions']]
    else:
        score = old['scores'][method]
    row = describe(score, selected, old['keep_local_indices'], old['gold_eligible_local'], old_token)
    assert abs(row['recovery'] - old['metrics'][method]['recovery']) < 1e-6
    result['old_8B'][method] = row

user_positions = spans['cases'][0]['input_metadata']['author_user_positions']
gold = set(mapping['gold_user_token_indices']) & set(mapping['keep_local_indices'])
for method, original in quality['methods'].items():
    row = describe(vectors[method][user_positions], original['selected_user_indices'],
                   mapping['keep_local_indices'], gold, new_token)
    assert row['recovery'] == original['recovery']
    result['new_9B'][method] = row

for b, case in enumerate(spans['cases']):
    lp0, lp1 = (np.asarray(x, dtype=np.float64) for x in root['endpoint_target_logprobs'][2*b:2*b+2])
    delta = lp1 - lp0
    lo, hi = case['mapping']['new_sink_span']
    result['target_effects'].append({
        'dataset': case['dataset'], 'index': case['index'], 'units': 'nats',
        'whole_response_delta': float(delta.sum()), 'thinking_delta': float(delta[:lo].sum()),
        'answer_delta': float(delta[lo:hi+1].sum()), 'trailing_EOS_delta': float(delta[hi+1:].sum()),
        'thinking_fraction_of_whole_delta': float(delta[:lo].sum() / delta.sum()),
    })

delta = np.asarray(root['endpoint_target_logprobs'][1]) - np.asarray(root['endpoint_target_logprobs'][0])
occurrences = []
for number in record['metadata']['outputs']:
    for occurrence, match in enumerate(re.finditer(re.escape(number), record['target'])):
        indices = [i for i, (a, b) in enumerate(new_token['generation_offsets'])
                   if a < match.end() and b > match.start()]
        occurrences.append({'number': number, 'occurrence': occurrence + 1,
                            'target_offsets': indices, 'delta': float(delta[indices].sum()),
                            'in_answer': min(indices) >= mapping['new_sink_span'][0]})
result['NI_number_occurrences'] = occurrences
result['NI_first_number_mentions_delta'] = sum(r['delta'] for r in occurrences if r['occurrence'] == 1)
result['NI_answer_numbers_delta'] = sum(r['delta'] for r in occurrences if r['in_answer'])
result['NI_answer_numbers_fraction_of_answer_delta'] = result['NI_answer_numbers_delta'] / result['target_effects'][0]['answer_delta']
result['NI_answer_token_effects'] = [
    {'offset': i, 'token': new_token['generation_tokens'][i], 'delta': float(delta[i])}
    for i in range(mapping['new_sink_span'][0], mapping['new_sink_span'][1]+1)
]

# Fixed FT0 layer diagnostics, without selecting/combining layers into a method.
item = quality['protocol']['score_files']['FT_corrected_native_hop0']
layers = read('snapshot' + item['path'], item['sha256'], numpy=True)['layer_token_scores'][:, 0]
keep_abs = np.asarray(user_positions)[mapping['keep_local_indices']]
gold_abs = sorted(set(mapping['gold_formatted_input_indices']) & set(keep_abs))
question_abs = sorted(set(keep_abs) & {user_positions[j] for j in new_token['question_local_positions']})
result['FT0_layer_diagnostics'] = []
for i, vector in enumerate(layers):
    vector = vector.astype(np.float64)
    mass = vector[keep_abs].sum()
    result['FT0_layer_diagnostics'].append({
        'layer': i, 'kind': 'FA' if i % 4 == 3 else 'GDN',
        'eligible_mass': float(mass),
        'gold_mass_fraction': float(vector[gold_abs].sum() / mass),
        'question_mass_fraction': float(vector[question_abs].sum() / mass),
    })
result['interpretation'] = {
    'accepted_failure': 'Current DT and corrected-native FT0-3 fail the first NI0 evidence-recovery acceptance. Operator provenance/conservation do not override this result.',
    'established_target_confound': 'Old P1 seeds the whole fixed response; new P1 seeds the answer conditional on unchanged thinking. Both contain the answer numbers repeatedly. Saved endpoint effects show the final numeric-answer target barely changes after eligible-prompt replacement.',
    'mechanism_hypothesis': 'Conditioning on evidence already copied into fixed thinking suppresses direct prompt dependence of the final numbers; remaining answer effect emphasizes wording. This explains a serious DT target mismatch, not the entire FT failure.',
    'FT_boundary': 'FT uses representation contributions and a thinking-hop heuristic, not this scalar logprob target. Prompt-tail concentration also occurs in FA layers; do not assign all failure to FLA or assume the DT diagnosis proves the FT cause.',
    'not_proved': 'No controlled old/new architecture-only attribution comparison, no all-dataset failure theorem, no recovered quality advantage, and no deletion-sign claim.',
    'next_decisive_control': 'Freeze at most one B2 whole-response P1 regression control using the same saved native endpoints and existing finite rules, then original CPU needle only. Preserve answer-only result and unchanged FT0-3. This diagnoses the migration target change; it is not permission to replace the official answer goal or tune targets using gold.',
    'metrics_decision': 'Defer the unstarted 97-forward RISE/MAS screen until this shared failure and target definition are resolved.',
}
result['source_sha256'] = receipts
path = A / 'qwen35_shared_failure_summary_20260908.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'status': result['status'],
    'new_question_top31': {k: v['selected_in_final_question'] for k, v in result['new_9B'].items()},
    'number_first_vs_answer_effect': [result['NI_first_number_mentions_delta'], result['NI_answer_numbers_delta']],
    'target_effects': result['target_effects'], 'new_model_forwards': 0,
}, ensure_ascii=False))
