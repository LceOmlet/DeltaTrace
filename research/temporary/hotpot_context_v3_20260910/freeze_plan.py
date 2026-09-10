"""Freeze full-input identities and v3 choices before the new GPU execution."""
import json
from artifacts_v3 import HERE,ROOT,V2,OLD,full_cases,sha

def main():
    full,receipts=full_cases()
    inputs=[]
    for i,(row,_) in sorted(full.items()):
        inputs.append(dict(index=i,input_sha256=row['input_sha256'],reference_sha256=row['references']['full'],
            original_target_sha256=row['original_target_sha256'],original_answer_token_span=row['original_answer_token_span'],
            prompt_length=row['prompt_length'],target_length=row['target_length'],full_target_weights=row['target_weights']))
    oldplan=json.loads((V2/'protocol.json').read_bytes())
    frozen=dict(cases=inputs,origins=receipts,full_analysis_sha256=oldplan['prior_full_analysis_sha256'])
    path=HERE/'full_input_identity.json'
    path.write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    plan=dict(version='hotpot-context-cost-v3',case_count=48,primary_labels='official_restored',
        spec_sha256=sha(HERE/'PROTOCOL.md'),source_sha256=oldplan['source_sha256'],
        tasks={'hotpotqa_long':oldplan['tasks']['hotpotqa_long']},
        chunks={f'conditioned_{k:02d}':list(range(16*k,16*(k+1))) for k in range(3)},
        weighted_sources=oldplan['weighted_sources'],FT_initial_target_adapter_sha256=oldplan['FT_initial_target_adapter_sha256'],
        answer_control_results_sha256=oldplan['answer_control_results_sha256'],answer_control_vectors_sha256=oldplan['answer_control_vectors_sha256'],
        input_identity_sha256=sha(path),prior_full_analysis_sha256=oldplan['prior_full_analysis_sha256'],
        attribution_module_sha256=sha(ROOT/'experiments/official/hotpot_retrieval_v3.py'),
        attribution_targets=['full','answer_conditioned'],pooling=['signed_sum','positive_mean_eligible'],
        body_token_budgets=[.05,.1,.2],sentence_budgets=[2,4,8],packing='longest_affordable_rank_prefix',
        primary_comparisons=4,bootstrap_draws=10000,bootstrap_seed=73,adjusted_confidence=.9875,
        independent_holdout=False,requires_dt_win=False)
    (HERE/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=48,protocol_sha256=sha(HERE/'protocol.json'),input_sha256=sha(path))))

if __name__=='__main__':main()
