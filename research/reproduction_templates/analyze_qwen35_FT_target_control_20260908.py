"""Verify FT target diagnostic and identify the old/new controller difference."""
import ast,hashlib,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT_target_control_20260908_v1';sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
assert sha((D/'review_bundle.zip').read_bytes())==receipt['bundle_sha256']
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        f=(D/name).resolve();assert f.is_relative_to(D.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(name))
r=json.loads((D/'results.json').read_bytes());p=r['protocol']
assert sha((D/'results.json').read_bytes())==receipt['results_sha256']
assert sha((D/'protocol.json').read_bytes())==sha((A/'qwen35_FT_target_control_protocol_20260908.json').read_bytes())
assert r['status']=='one_FT_whole_response_target_diagnostic_executed'
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['content_calls']==r['aggregate_calls']==r['parameter_cache_loads']==32
assert r['full_model_loads']==r['root_forwards']==r['decoder_replays']==r['generation_calls']==0
old=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT32_resume_20260908_v1/results.json').read_bytes())
assert r['input_cache_receipts']==old['input_cache_receipts'] and r['sources_before']==old['sources_before']
for name,digest in p['files_sha256'].items():
    raw=(D/name).read_bytes();assert sha(raw)==digest;ast.parse(raw)
    if name!='study.py':assert digest==old['protocol']['files_sha256'][name]
for item in r['artifacts']:assert sha((D/item['file']).read_bytes())==item['sha256']
values=dict(np.load(D/'whole_response_FT_diagnostic.npz',allow_pickle=False))
assert all(np.isfinite(x).all() for x in values.values())
total=np.zeros((2,605),dtype=np.float32)
for layer in values['layer_token_scores']:total+=layer
assert np.array_equal(total,values['token_total'])
assert np.array_equal(np.flatnonzero(values['weights'][0]),np.arange(357,605))
assert np.array_equal(np.flatnonzero(values['weights'][1]),np.arange(227,368))
assert not np.count_nonzero(total[1,368:])
spans=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json').read_bytes())['cases'][0]
keep=spans['mapping']['keep_local_indices'];gold=set(spans['mapping']['gold_user_token_indices'])&set(keep)
score=total[0,spans['input_metadata']['author_user_positions']]
top=np.asarray(keep)[np.argsort(-score[keep],kind='stable')[:31]].tolist()
q=r['original_NI0_recovery'];assert set(top)==set(q['selected_user_indices']) and len(set(top)&gold)/len(gold)==q['recovery'] and q['cutoff_ties']==1
tokens=json.loads((A/'snapshot/tmp/qwen35_shared_failure_tokens_20260908.json').read_bytes())['cases'][1]
question=set(tokens['question_local_positions'])
events=Counter(event['node'] for row in r['hops']['0']['layers'].values() for event in row['native_node_events'])
assert sum(v for k,v in events.items() if 'FlashAttn' in k)==8
assert sum(v for k,v in events.items() if 'GatedDelta' in k)==24
assert sum(v for k,v in events.items() if 'CausalConv' in k)==24
source_path=A/'snapshot${FLASHTRACE_ROOT}/ft_ifr_improve.py';source=source_path.read_bytes().replace(b'\r\n',b'\n')
assert sha(source)==p['author_metric_source_sha256']
cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='LLMIFRAttributionBoth')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='calculate_ifr_multi_hop_both')
segment=ast.get_source_segment(source.decode(),fn)
for marker in ['end_no_eos = int(gen_len) - 2','sink_start=all_gen_start_abs','base_total = base_total * stop_keep_mask_full','span_stops.append(is_stop_token(tok))']:
    assert marker in segment
history=A/'snapshot${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1/study.py'
history_text=history.read_text(encoding='utf-8')
assert "'attr_func':'ifr_multi_hop_both'" in history_text
summary={'status':'FT_target_diagnostic_8_of_40_and_controller_migration_difference_verified',
    'raw_sha256':receipt['results_sha256'],'bundle_sha256':receipt['bundle_sha256'],'protocol_sha256':sha((D/'protocol.json').read_bytes()),
    'source_sha256':p['files_sha256'],'same_native_caches_exact':True,'same_helpers_exact':True,
    'original_NI0_recovery':q,'selected_in_final_question':len(set(top)&question),
    'native_backward_node_counts':dict(events),'job_seconds_before_bundle':r['job_seconds_before_bundle'],
    'timed_call_seconds':sum(x['seconds'] for x in r['calls']),'peak_bytes_streamed_cache_path':max(x['peak_bytes'] for x in r['calls']),
    'budget':p['budget'],'all_jobs_terminal':True,'terminal_receipt':receipt,
    'controller_audit':{'author_both_source_sha256':sha(source),'author_both_method_lines':[fn.lineno,fn.end_lineno],
        'author_both_method_sha256':sha(segment.encode()),'old_development_study_sha256':sha(history.read_bytes()),
        'old_both':'All_gen(CoT+answer) for base and subsequent hops; excludes final EOS, stop-token sink/observation masking, output rows written only to answer span.',
        'current_9B_core':'Base answer span; later hops over thinking span; core controller does not implement original Both all_gen/stop-token policy.',
        'current_control':'One uniform whole-response span including EOS, no stop-token sink weighting; only a diagnostic, not a reproduction of Both.',
        'required_comparability_repair':'Restore the original Both0-3 controller as a separately named native9B comparator, keeping existing core0-3. Reuse original source/helper semantics and unchanged FA/FLA content kernels; no gold-based choices.'},
    'decision':'Target scope materially contributes to FT low NI0 recovery, but20% alone is not complete adaptation acceptance. The official Both family used in old experiments has not been carried over to the9B native comparison; restore this semantic gap before judging superiority.',
    'limits':['One historical case; original FT0-3 scores remain unchanged.','No RISE/MAS or independent confirmation.','Only target diagnostic, not a new official FT method or an architecture-only comparison.','Cold native calls,cache IO and streaming footprint are not production warm costs.']}
(A/'qwen35_FT_target_control_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','selected_in_final_question','native_backward_node_counts','job_seconds_before_bundle','timed_call_seconds']},ensure_ascii=False))
