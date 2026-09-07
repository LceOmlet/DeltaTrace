"""Prepare an unexposed original-source cohort without generating any answers."""
import ast,hashlib,json,typing
from dataclasses import dataclass,field
from pathlib import Path
A=Path(__file__).resolve().parent;B=A/'published_flashtrace/table1-data-v1'
raw_path=B/'extracted/data/with_human_verification.json';cache_path=B/'extracted/data/morehopqa.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(raw_path)=='41b1c31af2546f005fd699148bc1ae68968349179941f0218ef7975596489f4a'
assert sha(cache_path)=='c2bcc582f316b0774abea731d763cd8b9bf3a53bf29a29021910e3eab7036782'
rows=json.loads(raw_path.read_text());old=[json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
assert len(rows)==1118 and len(old)==95
upstream=A/'published_flashtrace/075e7e44ae4d5acd2ed76e0d2aced57107d02736/attribution_datasets.py'
assert sha(upstream)=='2d65e9b5c96dc2817dfd478f247f43f3552cf44900df3ba94bced1517505e2cf'
tree=ast.parse(upstream.read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['AttributionExample','AttributionDataset','MoreHopQAAttributionDataset']]
# Run the exact original dataset classes, omitting only unrelated global NLP
# initialization/imports. No copied/rephrased prompt construction is used.
ns=dict(vars(typing),dataclass=dataclass,field=field,Path=Path,json=json)
exec(compile(ast.Module(body=classes,type_ignores=[]),'exact_original_MoreHopQA_classes','exec'),ns)
dataset=ns['MoreHopQAAttributionDataset'](raw_path)
byid={str(row['_id']):i for i,row in enumerate(rows)};assert len(byid)==len(rows)
old_indices=[]
for row in old:
    i=byid[str(row['metadata']['id'])];ex=dataset[i]
    assert ex.prompt==row['prompt'] and ex.metadata['original_context']==row['metadata']['original_context']
    assert ex.metadata['answer']==row['metadata']['reference_answer']
    old_indices.append(i)
assert len(set(old_indices))==95
canonical=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
used_ids={str(rows[i]['_id']) for i in old_indices}
used_context={canonical(rows[i]['context']) for i in old_indices}
used_base_questions={canonical(rows[i]['previous_question']) for i in old_indices if rows[i].get('previous_question')}
used_prompts={hashlib.sha256(r['prompt'].encode()).hexdigest() for r in old}
included=[];excluded=[]
for i,row in enumerate(rows):
    reasons=[]
    if str(row['_id']) in used_ids:reasons.append('same_original_id')
    if canonical(row['context']) in used_context:reasons.append('same_original_context')
    if row.get('previous_question') and canonical(row['previous_question']) in used_base_questions:reasons.append('same_base_question')
    if hashlib.sha256(dataset[i].prompt.encode()).hexdigest() in used_prompts:reasons.append('same_formatted_prompt')
    if reasons:excluded.append({'index':i,'reasons':reasons})
    else:included.append(i)
assert included and not set(included)&set(old_indices)
folder=A/'morehop_original_confirmation_20260907';folder.mkdir(exist_ok=True)
selected=folder/'eligible_original_source.json';selected.write_text(json.dumps([rows[i] for i in included],ensure_ascii=False,indent=2),encoding='utf-8')
restored=ns['MoreHopQAAttributionDataset'](selected)
assert len(restored)==len(included) and all(a.prompt==dataset[i].prompt and a.metadata==dataset[i].metadata for a,i in zip(restored,included))
summary={'status':'source_cohort_prepared_no_generation','user_authorization':'2026-09-07: allowed supplementing unused MoreHopQA by the original author sampling procedure, clearly distinguished from fixed Table1 reproduction.',
    'release_url':'https://github.com/wbopan/flashtrace/releases/tag/table1-data-v1','original_source_sha256':sha(raw_path),'original_cache_sha256':sha(cache_path),
    'original_source_rows':1118,'previously_used_cache_rows':95,'matched_old_original_indices':old_indices,
    'excluded_count':len(excluded),'excluded':excluded,'eligible_count':len(included),'eligible_original_indices':included,
    'eligible_source_sha256':sha(selected),'exact_original_prompt_and_metadata_verified':True,
    'exclusion_scope':'Remove original ID, whole original context, previous/base question, or formatted prompt overlap with all95 released historical MoreHopQA cases. Preserve remaining original row order and each original record unchanged.',
    'sampling_source_sha256':'de09bf82b8a710c5ed2fc383b7c954a073054440febaff1b37ab3cc2f6c9057e',
    'planned_generation_model':'qwen3-235b-a22b-2507','planned_judge_model':'deepseek-v3-1-terminus',
    'planned_kept_examples':64,'planned_max_raw_attempts':256,'generation_or_judge_calls':0,'attribution_or_metric_calls':0,
    'candidate_frozen_source_protocol':'vendor_fa_batch_memory_protocol_20260907.json',
    'limits':'Prepared source cohort only. No API generation, no new fixed trajectory cache, no independent attribution results. New sample selection is explicitly outside the published95-case Table1 reproduction.'}
(A/'original_morehop_confirmation_source_summary_20260907.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['matched_old_original_indices','excluded','eligible_original_indices']},indent=2))
