"""Freeze the original API sampling protocol; do not execute generation."""
import ast,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;folder=A/'morehop_original_confirmation_20260907'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((A/'original_morehop_confirmation_source_summary_20260907.json').read_text())
source=folder/'eligible_original_source.json';assert sha(source)==s['eligible_source_sha256']
rows=json.loads(source.read_text());assert len(rows)==s['eligible_count']==823
selected=folder/'sampling_original_source.json';selected.write_text(json.dumps(rows[:256],ensure_ascii=False,indent=2),encoding='utf-8')
study=A/'original_morehop_confirmation_sampler_20260907.py';ast.parse(study.read_text())
old=json.loads((A/'vendor_fa_batch_memory_protocol_20260907.json').read_text())
pins=dict(old['official_normalized_sources']);pins.update({'attribution_datasets.py':'2d65e9b5c96dc2817dfd478f247f43f3552cf44900df3ba94bced1517505e2cf','exp/exp2/sample_and_filter.py':'de09bf82b8a710c5ed2fc383b7c954a073054440febaff1b37ab3cc2f6c9057e'})
p={'status':'frozen_before_any_new_generation_or_attribution','purpose':'User-authorized new independent MoreHopQA confirmation source under the author original sampling process. Distinct from released95-case Table1 reproduction.',
    'original_repository':'${FLASHTRACE_ROOT}','original_normalized_sources':pins,'study_sha256':sha(study),
    'original_source_cohort_summary_sha256':sha(A/'original_morehop_confirmation_source_summary_20260907.json'),
    'sampling_source_sha256':sha(selected),'sampling_original_indices':s['eligible_original_indices'][:256],
    'maximum_raw_examples':256,'kept_examples_target':64,'maximum_API_attempts':1536,
    'generator_model':'qwen3-235b-a22b-2507','judge_model':'deepseek-v3-1-terminus','tokenizer_model':old['checkpoint'],
    'cache_namespace':'deltatrace-morehop-independent-20260907-v1','API_key_policy':'Environment only; never argv, committed file, audit record or chat.',
    'sampling_rules':'Unmodified original sample_and_filter.main: original order, first64 accepted; at most first256 unexposed source rows. Original prompts,8192-token generation limit,temperature0,two retries,64-token judge,original format/judge/span filters. Every failed/filtered attempt retained, no attribution-based filtering.',
    'method_selection_policy':'Finite P1 implementation unchanged and pinned by vendor_fa_batch_memory_protocol_20260907.json. No new-cohort scores used for tuning. Full comparison protocol and numerical/cost/sign claims must be frozen before attribution or original quality evaluation.',
    'uncertainty_policy':'Report original context/base-question grouping and the number of distinct source groups; use grouped paired uncertainty rather than treating related questions as independent observations. Uncertainty is reported, not used for post-hoc parameter tuning.',
    'completion_policy':'Sampling is not independent quality confirmation. Freeze and independently validate resulting complete targets/spans/IDs/API provenance before candidate or FT evaluation. If64 are not obtained, retain all outcomes and report sample-count shortfall, without cherry-picking.'}
(A/'original_morehop_sampler_protocol_20260907.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
print(json.dumps({'status':p['status'],'prepared_original_rows':256,'eligible_original_rows':823,'kept_target':64,'maximum_API_attempts':1536,'actual_API_calls':0,'source_bytes':selected.stat().st_size,'sampling_source_sha256':sha(selected)},indent=2))
