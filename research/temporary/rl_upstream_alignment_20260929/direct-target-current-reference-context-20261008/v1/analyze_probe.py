"""Describe the current actual token's four native endpoints and interactions."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
REPO=HERE.parents[4]
spec=importlib.util.spec_from_file_location('context_analysis_owner',AUDIT/'direct-target-reference-interaction-20261007/v1/analyze_reference_context.py')
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

folder=HERE/'actual-results'
for item in json.loads((folder/'transport.json').read_bytes()):
    assert binding(Path(item['local_path']))['sha256']==item['sha256']
case=json.loads((HERE/'case.json').read_bytes())
ranks=[json.loads((folder/f'results/rank{i}.json').read_bytes()) for i in (0,1)]
pairs=[owner.summarize_pair(folder/f'results/rank{i}-target-logp.pt',case['row'],case['packed_slot']) for i in (0,1)]
assert pairs[0]==pairs[1]
assert all(r['phase']=='complete' and r['native_sha256']==case['native_sha256'] and r['source_sha256']==case['source_sha256'] for r in ranks)
assert all(not shard['nonzero'] for r in ranks for shard in r['lora_B_local_shards'])
row=case['row']
old_path=AUDIT/'direct-target-credit-sample-20261007/v1/results-appworld/appworld-rank0-most_negative-targets.pt'
old=torch.load(old_path,map_location='cpu',weights_only=False)
assert old['native_sha256']==case['native_sha256'] and old['source_sha256']==case['source_sha256']
point=old['points'][row]
assert point['packed_slot']==case['packed_slot'] and point['token_id']==case['token_id']
F=float(old['factual_joint_logp'][row])
D=float(old['reference_joint_logp'][row])
B=ranks[0]['reference_joint_logp'][row]
C=ranks[0]['token_restored_joint_logp'][row]
curve_path=AUDIT/'direct-target-existing-pv-rule-20261008/v1/memory-curve-results/results/rank0.json'
curve=json.loads(curve_path.read_bytes())
assert curve['geometry']['candidate']['traj_uid']==case['uid']
curve_points=curve['views']['signed_RISE']['score_points']
result=dict(status='Current extreme token native four-endpoint diagnosis complete; no credit repair deployed',
    case=case,launch=json.loads((HERE/'launch.json').read_bytes()),
    endpoints=dict(F_factual_all_sources=F,D_factual_delete_only_candidate=D,
        B_all_prior_sources_EOS=B,C_only_candidate_restored=C),
    factual_context_effect=F-D,all_EOS_context_effect=C-B,
    measured_interaction_difference=(F-D)-(C-B),
    double_reduction_order_difference=(C-B)-pairs[0]['signed_future_effect'],
    native_pair_detail=pairs[0],original_native_owners=[r['owners'] for r in ranks],
    original_joint_DT_point=json.loads((REPO/'experiments/rl/results_credit_causal_diagnosis_20261008.json').read_bytes())['selected_actual_tokens']['AppWorld_newline'],
    exact_endpoint_controls=dict(ranks_same_per_target_values=True,other_three_pair_effects_zero=True,
        earlier_target_delta_zero=True,lora_B_shards_all_zero=True,
        other_three_background_scores_minus_previous_factual=[ranks[0]['reference_joint_logp'][i]-float(old['factual_joint_logp'][i]) for i in range(4) if i!=row],
        F_minus_previous_actual_author_curve=F-curve_points[0]['logp'],
        B_minus_previous_actual_author_curve=B-curve_points[-1]['logp']),
    measurements=dict(native_paired_forward_calls_per_rank=1,DT=0,rollout=0,backward=0,optimizer=0,checkpoint_restore=0,
        native_forward_seconds=[r['seconds'] for r in ranks],recorded_pss_bytes=[r['pss_bytes'] for r in ranks],
        continuous_physical_peak_measured=False),
    source_bindings=[binding(old_path),binding(curve_path),binding(folder/'transport.json'),
        binding(HERE/'case.json'),binding(AUDIT/'direct-target-reference-interaction-20261007/v1/inspect_reference_context.py'),
        binding(REPO/'experiments/rl/upstream.lock')],
    scope='Same current AppWorld newline at packed2883, original real B4 and full scattered joint action Y. One native forward supplies all-EOS and only-candidate-restored endpoints; factual and factual-single-delete endpoints come from the previously recorded same native input source. Descriptive interaction measurement, not an official tolerance or a new estimator.',
    limitations=['Reference-context effects are not substituted into production d/Q/V/A.',
        'The four-endpoint interaction is not an independent additive token credit or a population error rate.',
        'Original-author cumulative deletion/RISE/MAS remain linked separately and are not replaced by this selected-token diagnosis.',
        'The original snapshot field names reference/factual refer to paired first/second rows; here the second row restores only this source in the all-EOS context, not the original full factual history.'],
    official_tolerance_claim=False,credit_repaired=False,production_profile_changed=False,formal_restart=False,TextCraft_update_released=False,
    unchanged=dict(QVA=True,PPO=True,whitening=True,target=True,reward=True,lora_rank=8,lora_alpha=16,actor_microbatch_per_GPU=4,DT_minibatch_per_GPU=4))
target=REPO/'experiments/rl/results_current_token_reference_context_20261008.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(HERE/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not torch.cuda.is_initialized()
print(json.dumps({key:result[key] for key in ['endpoints','factual_context_effect','all_EOS_context_effect','measured_interaction_difference','exact_endpoint_controls','measurements']},ensure_ascii=False))
