"""Inspect one saved DT source token and endpoint scores; no model evaluation."""
import hashlib,json,math,os,sys,time
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-efficiency-20261015-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
launch=json.loads((RUN/'launch.json').read_bytes());os.environ.update(launch['effective_environment']);os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=launch['effective_environment']['PYTHONPATH'].split(':')
import torch
from transformers import AutoTokenizer
torch.set_num_threads(2)
audit=json.loads((DEST/'step24-saved-credit-CPU-audit.json').read_bytes())
case=audit['steps']['24']['raw_minimum'];record=torch.load(case['source'],map_location='cpu',weights_only=True)
row=next(v for v in record['rows'] if v['traj_uid']==case['traj_uid'])
tokenizer_path=launch['options']['actor_rollout_ref.model.path']
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
pos=case['raw_min_response_position'];suffix=row['suffix_positions'];selected_index=case['prompt_length']+int((suffix==pos).nonzero().item())
ids=row['input_ids'];pol=row['policy_mask'];targets=row['target_mask'];ratios=row['source_log_ratios']
negative=[int(i) for i in torch.argsort(ratios.masked_fill(~pol,float('inf')))[:10]]
def describe(index):
    full_index=case['prompt_length']+int((suffix==index).nonzero().item())
    return dict(response_position=index,selected_input_position=full_index,token_id=int(ids[full_index]),
        token_text=tokenizer.decode(ids[full_index:full_index+1],skip_special_tokens=False),
        surrounding_text=tokenizer.decode(ids[max(0,full_index-28):min(len(ids),full_index+29)],skip_special_tokens=False),
        DT_log_ratio=float(ratios[index]),policy=bool(pol[index]),target=bool(targets[index]),prior_source=bool(row['prior_source_mask'][index]),
        original_token_equals_EOS=int(ids[full_index])==record['eos_token_id'])
target_spans=[]
for offset in row['target_offsets']:
    if not target_spans or offset!=target_spans[-1][-1]+1:target_spans.append([offset])
    else:target_spans[-1].append(offset)
target_data=[dict(response_start=s[0],response_end=s[-1]+1,text=tokenizer.decode(ids[case['prompt_length']+s[0]:case['prompt_length']+s[-1]+1],skip_special_tokens=False)) for s in target_spans]
lp=row['factual_target_logp'];d=case['d_at_min'];g=case['reward_multiplier']
value=dict(unix=time.time(),scope=__doc__,source=case['source'],source_SHA256=hashlib.sha256(Path(case['source']).read_bytes()).hexdigest(),
    trajectory_UID=case['traj_uid'],source_code_binding=record['readout_source'],tokenizer_path=tokenizer_path,
    actual_context_tokens=len(ids),original_EOS_token_id=record['eos_token_id'],
    extreme_source_token=describe(pos),most_negative_source_tokens=[describe(i) for i in negative],
    executed_target_spans=target_data,target_token_count=len(row['target_offsets']),
    factual_joint_target_logp=lp,reference_all_source_EOS_joint_target_logp=row['reference_target_logp'],
    probability_consistency=dict(factual_probability=math.exp(lp),DT_d=d,DT_implied_deleted_log_probability=lp-d,
        DT_implied_deleted_probability=math.exp(lp-d),largest_valid_probability_ratio=math.exp(-lp),
        DT_implied_probability_ratio=math.exp(-d),
        advantage_lower_bound_if_d_is_exact_single_token_deletion_ratio=g*(-math.expm1(-lp)),
        observed_raw_advantage=case['raw_min'],
        statement='An individual token deletion probability cannot exceed1. This rules out interpreting this saved DT estimate as an exact deletion log-ratio; it does not identify a particular finite operator defect.',
        new_numerical_tolerance=None),
    model_forward_backward_DT_optimizer_calls=0,training_changes=0)
(DEST/'step24-extreme-source-token-CPU.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(value,ensure_ascii=False))
