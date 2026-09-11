"""Freeze requested task scopes and remaining baseline settings before GPU scores."""
import json
from common import HERE,ROOT,OLD,HOTPOT,TASKS,BASELINES,sha

def main():
    audit=json.loads((HERE/'input_audit.json').read_bytes());assert audit['case_count']==448
    for name,h in audit['file_sha256'].items():assert sha(HERE/name)==h
    source=json.loads((HERE/'baseline_source_identity.json').read_bytes())
    mlm=json.loads((HERE/'mlm_identity.json').read_bytes())
    plan=dict(version='all-baselines-fixed-evidence-v1',status='frozen_before_remaining_baseline_quality',
        selection_history='HotpotQA full signed-sum selected by user after DT/FT results; VT retains previous fixed policy. Retrospective.',
        spec_sha256=sha(HERE/'PROTOCOL.md'),tasks={t:dict(count=48 if t=='hotpotqa_long' else 100,
            target_mode='full' if t=='hotpotqa_long' else 'answer_only',cache_sha256=audit['cache_sha256'][t]) for t in TASKS},
        new_methods=list(BASELINES),reuse_methods=['DT','FT_K3','FT_K1'],new_method_cases=2240,
        metrics=['recall'],fractions=[.05,.1,.2],hotpot_pooling='signed_sum',hotpot_selection='all_body_tokens_prefix',
        hotpot_sentence_budgets=[2,4,8],vt_primary_view='raw',vt_diagnostics=['density'],
        target_aggregation='raw_signed_sum_with_saved_target_weights',native_positive_row_normalized_diagnostic=True,
        source_k=20,ifr_chunk_tokens=128,ifr_sink_chunk_tokens=32,attnlrp_score_mode='generated',
        input_audit_sha256=sha(HERE/'input_audit.json'),inputs_sha256=sha(HERE/'inputs.json'),
        candidates_sha256=sha(HERE/'candidates.json'),labels_sha256=sha(HERE/'labels.json'),
        source_identity_sha256=sha(HERE/'baseline_source_identity.json'),
        baseline_sources={k:v['normalized_sha256'] for k,v in source['files'].items()},
        mlm=dict(repository=mlm['repository'],revision=mlm['revision'],files=[{k:v for k,v in r.items() if k!='pending_user_download'} for r in mlm['files']]),
        previous_full_analysis_sha256=sha(OLD/'full_recall/analysis.json'),hotpot_analysis_sha256=sha(HOTPOT/'analysis.json'),
        hotpot_protocol_sha256=sha(HOTPOT/'protocol.json'),vt_choice_sha256=sha(OLD/'scope_choice.json'),
        pilot=[dict(dataset=t,index=0) for t in TASKS],seed=73,
        bootstrap=dict(draws=10000,seed=73,paired=True,vt_stratified_by_task=True,primary_family=12,confidence=1-.05/12,descriptive=True))
    path=HERE/'protocol.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=plan['status'],protocol_sha256=sha(path),new_method_cases=2240)))

if __name__=='__main__':main()
