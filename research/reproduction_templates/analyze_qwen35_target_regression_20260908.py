"""Verify the single target-control result and its frozen source/cost boundary."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
D = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_target_regression_20260908_v1'
P = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1'
sha = lambda b: hashlib.sha256(b).hexdigest()
receipt = json.loads((D / 'terminal_receipt.json').read_bytes())
assert receipt['proc_exists'] is False
assert sha((D / 'review_bundle.zip').read_bytes()) == receipt['bundle_sha256']
with zipfile.ZipFile(D / 'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        f = (D / name).resolve(); assert f.is_relative_to(D.resolve())
        f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(z.read(name))
r = json.loads((D / 'results.json').read_bytes()); p = r['protocol']
old = json.loads((P / 'results.json').read_bytes())
assert sha((D / 'results.json').read_bytes()) == receipt['results_sha256']
assert sha((D / 'protocol.json').read_bytes()) == sha((A / 'qwen35_target_regression_protocol_20260908.json').read_bytes())
assert r['status'] == 'one_whole_response_target_regression_executed'
assert sha((P / 'results.json').read_bytes()) == p['regression_parent_results_sha256']
assert r['sources_before'] == r['sources_after'] == old['sources_before']
assert r['weight_stats_before'] == r['weight_stats_after'] == old['weight_stats_before']
assert r['weight_tensor_receipts'] == old['weight_tensor_receipts']
for name, digest in p['files_sha256'].items():
    raw = (D / name).read_bytes(); assert sha(raw) == digest; ast.parse(raw)
    if name != 'study.py': assert digest == old['protocol']['files_sha256'][name]
for item in r['artifacts']:
    if item['download']:
        raw = (D / item['file']).read_bytes(); assert sha(raw) == item['sha256'] and len(raw) == item['bytes']
assert r['root_forwards'] == r['full_model_loads'] == r['generation_calls'] == r['head_norm_backwards'] == 0
assert r['finite_decoder_calls'] == r['decoder_loads'] == 32
assert r['decoder_replays'] == 30 and r['decoder_capture_reuses'] == 2
assert r['auxiliary_FA_calls'] == 7 and r['finite_answer_calls'] == r['head_norm_forwards'] == r['original_recovery_calls'] == 1
assert [x['count'] for x in r['target_selection']] == [248,141]
assert all(x['offsets'] == list(range(x['count'])) for x in r['target_selection'])

whole = dict(np.load(D / 'whole_input_review.npz', allow_pickle=False))
prior = dict(np.load(P / 'whole_input_review.npz', allow_pickle=False))
seed = dict(np.load(D / 'target_seed_review.npz', allow_pickle=False))
for name in ['input0', 'input1']: assert np.array_equal(whole[name], prior[name])
for array in list(whole.values()) + list(seed.values()): assert np.isfinite(array).all()
allocation = lambda x: (x['finite'].astype(np.float64) * (x['input1'].astype(np.float64)-x['input0'].astype(np.float64))).sum(-1)
signed = allocation(whole); sums = signed.sum(1); head_sum = allocation(seed).sum(1)
assert np.max(np.abs(sums - r['signed_sums'])) < 1e-7
assert np.max(np.abs(head_sum - r['head_and_norm_finite_effect'])) < 1e-7
assert np.max(np.abs(signed - whole['signed'])) < 1e-5
assert np.count_nonzero(whole['finite'][1,368:]) == 0

root = np.asarray(r['original_root_target_logprobs']).reshape(-1,2)
replay = np.asarray(r['replayed_target_logprobs']).reshape(-1,2)
cuts = [0,248,389]
sum_samples = lambda x: np.array([x[cuts[b]:cuts[b+1]].sum() for b in range(2)])
root_delta = sum_samples(root[:,1]-root[:,0]); replay_delta = sum_samples(replay[:,1]-replay[:,0])
assert np.max(np.abs(root_delta - r['original_root_target_delta'])) < 1e-8
assert np.max(np.abs(replay_delta - r['head_replay_delta'])) < 1e-8
old_answer = np.asarray(old['replayed_answer_logprobs']).reshape(-1,2)
answer_rows = [j + cuts[b] for b, row in enumerate(old['target_selection']) for j in row['offsets']]
answer_logprob_diff = float(np.max(np.abs(replay[answer_rows] - old_answer)))
last = head_sum; local = []; discontinuity = []
for i in reversed(range(32)):
    row = r['layers'][str(i)]
    before, actual, after = [np.asarray(row[k]) for k in ['root_output_effect','replay_output_effect','input_effect']]
    assert np.max(np.abs(before-last)) < 1e-7 and row['padding_max_abs'] == 0
    local.append(after-actual); discontinuity.append(actual-before); last = after
assert np.max(np.abs(last-sums)) < 1e-7
ledger = (replay_delta-root_delta) + (head_sum-replay_delta) + np.sum(local,axis=0) + np.sum(discontinuity,axis=0)
assert np.max(np.abs(ledger - (sums-root_delta))) < 1e-7

spans = json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json').read_bytes())['cases']
mapping = spans[0]['mapping']; user = spans[0]['input_metadata']['author_user_positions']
keep = mapping['keep_local_indices']; gold = set(mapping['gold_user_token_indices']) & set(keep)
selected = r['original_NI0_recovery']['selected_user_indices']
scores = whole['signed'][0,user].astype(np.float64); positive = np.maximum(scores,0)
top = np.asarray(keep)[np.argsort(-positive[keep],kind='stable')[:31]].tolist()
assert set(selected) == set(top) and len(set(selected)&gold)/len(gold) == r['original_NI0_recovery']['recovery']
assert r['original_NI0_recovery']['cutoff_ties'] == 1
tokens = json.loads((A/'snapshot/tmp/qwen35_shared_failure_tokens_20260908.json').read_bytes())['cases'][1]
question = set(tokens['question_local_positions']); denominator = positive[keep].sum()
localization = {
    'selected_in_question': len(set(selected)&question), 'gold_positive_count': int((scores[sorted(gold)]>0).sum()),
    'gold_negative_count': int((scores[sorted(gold)]<0).sum()),
    'gold_positive_mass_fraction': float(positive[sorted(gold)].sum()/denominator),
    'question_positive_mass_fraction': float(positive[sorted(question&set(keep))].sum()/denominator),
    'top_ten_token_text': [tokens['user_tokens'][j] for j in selected[:10]],
}
costs = {}
for call in r['calls']:
    kind = call['kind']
    group = ('weight_load' if kind.startswith('load_original_') else 'saved_capture_load' if kind.startswith('load_saved_') else
             'finite_FA_decoder' if kind.startswith('finite_decoder') and int(kind.removeprefix('finite_decoder'))%4==3 else
             'finite_GDN_decoder' if kind.startswith('finite_decoder') else 'auxiliary_FA' if 'public_FA_auxiliary' in kind else
             'original_decoder_replay' if kind.startswith('original_decoder') else 'target_and_norm')
    item = costs.setdefault(group, {'calls':0,'seconds':0.0,'absolute_peak_bytes':0})
    item['calls'] += 1; item['seconds'] += call['seconds']; item['absolute_peak_bytes'] = max(item['absolute_peak_bytes'],call['peak_bytes'])
summary = {
    'status':'same_endpoint_target_control_restores_NI0_recovery_to_16_of_40',
    'raw_sha256':receipt['results_sha256'],'bundle_sha256':receipt['bundle_sha256'],
    'protocol_sha256':sha((D/'protocol.json').read_bytes()),'source_sha256':p['files_sha256'],
    'same_native_endpoints_exact':True,'same_native_weight_receipts_exact':True,'same_runtime_sources_exact':True,
    'old_answer_logprob_max_abs_difference_under_new_head_packing':answer_logprob_diff,
    'root_vs_replayed_logprob_max_abs':float(np.max(np.abs(root-replay))),
    'target_selection':r['target_selection'],'original_NI0_recovery':r['original_NI0_recovery'],
    'NI0_localization':localization,'root_target_delta':root_delta.tolist(),'signed_sums':sums.tolist(),
    'relative_total_residual':((sums-root_delta)/np.abs(root_delta)).tolist(),
    'positive_mass':np.maximum(signed,0).sum(1).tolist(),'negative_mass':np.minimum(signed,0).sum(1).tolist(),
    'ledger_reconciles':True,'sum_absolute_local_residual':np.abs(local).sum(0).tolist(),
    'sum_absolute_replay_discontinuity':np.abs(discontinuity).sum(0).tolist(),
    'costs':costs,'job_seconds_before_final_zip':r['job_seconds'],
    'unsegmented_seconds':r['job_seconds']-sum(x['seconds'] for x in r['calls']),
    'compiler_benchmark_observations':r['compiler_benchmark_observations'],'compiler_counters':r['compiler_counters'],
    'budget':p['budget'],'terminal_receipt':receipt,'all_jobs_terminal':True,
    'decision':'The DT target migration is a major cause of this NI0 regression:1/40 to16/40 with unchanged finite rules/endpoints, near historical old8B17/40. Restore an explicit whole-response regression reference; the answer-only version remains a failed evidence-recovery candidate. FT source-recovery failure remains unresolved and its weakness cannot establish useful superiority.',
    'limits':['One historical NI0 control, not independent confirmation or paper-table reproduction.',
              'MH1 attribution computed in the same B2 but RISE/MAS remain unexecuted; no MH quality claim.',
              'Whole-response logprob and FT answer-representation/hop internals are distinct, even on the same official evidence task.',
              'Signed finite allocation is not a conditional deletion-sign proof.',
              'Streamed weight loads,cold replay,diagnostic and IO costs are paid; not warm full-model speed or memory evidence.'],
}
(A/'qwen35_target_regression_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','old_answer_logprob_max_abs_difference_under_new_head_packing','root_vs_replayed_logprob_max_abs','NI0_localization','relative_total_residual','costs','job_seconds_before_final_zip']},ensure_ascii=False))
