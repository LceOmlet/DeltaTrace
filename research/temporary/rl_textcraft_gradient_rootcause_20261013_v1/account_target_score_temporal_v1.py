"""Read existing target probabilities and bound output-score gradients, no model."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from verl import DataProto

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
raw=json.loads((OUT/'temporal-native-reports-v1.json').read_bytes())
work=json.loads((OUT/'native-workload-temporal-v1.json').read_bytes())['rows']
readouts=[raw['raw_reports'][k] for k in sorted(raw['raw_reports'])]
assert len(readouts)==2 and len(work)==85
report={'unix':time.time(),'scope':__doc__,'model_GPU_backward_optimizer_calls':0,
        'temporal':[], 'limits':[
 'Temporal target probabilities are recorded factual DT/HF endpoints, calibrated separately against native old-logprobs at iteration81.',
 'Full-history token counts precede native retained-response mapping. Output-logit gradient bounds are not parameter-gradient norms.',
 'The temporal full-history equal-token mean is a diagnostic common normalization, not a replacement of the original four-minibatch/B4 training objective.',
 'For a trajectory with target NLL L, sum(1-p_token) is between1-exp(-L) and L. Thus target output-gradient L1 can be bounded without inventing individual probabilities.',
 'Zero-reward and positive-reward source gradients are analyzed from the existing full iteration81 snapshot, not fabricated for historical iterations.',
 ]}
per_step=[]
for i,m in enumerate(raw['metrics']):
    assert m['step']==i+1
    traces={}
    repeat_discrepancies=[]
    for rank in readouts:
        for tr in rank[i]['value']['traces']:
            uid=tr['traj_uid']
            if uid in traces:
                repeat_discrepancies.append(abs(tr['factual_target_logp']-traces[uid]['factual_target_logp']))
            else: traces[uid]=tr
    assert len(traces)==work[i]['nonzero_requests'],(i+1,len(traces),work[i]['nonzero_requests'])
    per_step.append(traces)
    a=m['values']['critic/advantages/max']
    n=work[i]['full_history_policy_tokens']
    bad=[t for t in traces.values() if t['factual_target_logp'] is None or t['factual_target_logp']>0]
    valid=[t for t in traces.values() if t not in bad]
    L=[-t['factual_target_logp'] for t in valid]
    low=sum(-math.expm1(-v) for v in L)
    high=sum(L)
    row={'step':i+1,'native_grad_norm':m['values']['actor/grad_norm'],
        'full_history_policy_tokens':n,'successful_trajectories':len(traces),
        'native_whitened_target_positive_coefficient_rounded':a,
        'full_success_target_tokens':sum(t['target_tokens'] for t in valid),
        'successful_target_NLL_sum':high,
        'successful_target_one_minus_probability_sum_bounds':[low,high],
        'full_history_common_token_mean_target_logit_gradient_L1_bounds':[2*a*low/n,2*a*high/n],
        'invalid_probability_records':bad,
        'padding_duplicate_endpoint_max_abs_difference':max(repeat_discrepancies,default=0),
        'sources':{'native_metrics':m['sources'],'workload':work[i]['source'],
            'readout':[rank[i]['source'] for rank in readouts]}}
    report['temporal'].append(row)

inputs=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
batch=DataProto.concat([DataProto.load_from_disk(r['input']) for r in inputs])
width=batch.batch['responses'].shape[-1]
mask=batch.batch['loss_mask'][:,-width:].bool()
target=torch.zeros_like(mask)
for i,art in enumerate(batch.non_tensor_batch['dt_direct_target_artifact']):
    positions=torch.as_tensor(art['retained_response_positions'])
    keep=positions>=0
    target[i,keep]=torch.as_tensor(art['target_mask'],dtype=torch.bool)[positions[keep]]
reward=batch.batch['token_level_rewards'].double().sum(-1)
success=reward!=0
a=batch.batch['advantages'].double()
lp=batch.batch['old_log_probs'].double()
oldp=lp.exp()
groups={
    'successful_target':target&success[:,None],
    'successful_source':mask&~target&success[:,None],
    'zero_reward_all_policy':mask&~success[:,None],
}
assert torch.equal(sum(g.long() for g in groups.values()),mask.long())
badp=mask&(oldp>1)
report['snapshot81_oldprob_gt_one_count']=int(badp.sum())
assert not badp.any()
coeff=torch.zeros_like(a)
for rank in (0,1):
    for mini in range(4):
        for micro in range(8):
            start=rank*128+mini*32+micro*4
            sl=slice(start,start+4)
            coeff[sl]=a[sl]*mask[sl]/(16*int(mask[sl].sum()))
rows=[]
traces=per_step[80]
for i,uid in enumerate(batch.non_tensor_batch['traj_uid']):
    uid=str(uid)
    if not success[i] or uid not in traces: continue
    native_count=int(target[i].sum())
    full_count=traces[uid]['target_tokens']
    native_nll=float(-lp[i,target[i]].sum())
    original_nll=-traces[uid]['factual_target_logp']
    rows.append({'uid':uid,'native_target_tokens':native_count,'full_target_tokens':full_count,
        'native_old_target_NLL':native_nll,'DT_factual_target_NLL':original_nll,
        'same_target_count':native_count==full_count,
        'NLL_difference_native_minus_DT':native_nll-original_nll})
report['snapshot81_target_probability_calibration']=rows
budget=[]
for mini in range(4):
    idx=torch.cat((torch.arange(mini*32,(mini+1)*32),torch.arange(128+mini*32,128+(mini+1)*32)))
    record={'minibatch_index':mini,'groups':{}}
    for name,g in groups.items():
        c=coeff[idx][g[idx]]
        p=oldp[idx][g[idx]]
        part=c*(1-p)
        lo=float(part.square().sum())**.5
        record['groups'][name]={
            'tokens':int(g[idx].sum()),'logprob_derivative_coefficient_L2':float(c.square().sum())**.5,
            'logit_gradient_L2_bounds':[lo,math.sqrt(2)*lo],
            'logit_gradient_L1_exact_from_selected_probability':2*float(part.abs().sum()),
            'selected_old_logprob_mean':float(lp[idx][g[idx]].mean()) if g[idx].any() else None,
        }
    budget.append(record)
report['snapshot81_original_B4_PG_output_budget']=budget
report['snapshot81_retained_policy_tokens']=int(mask.sum())
report['snapshot81_full_history_policy_tokens']=work[80]['full_history_policy_tokens']
report['input_sources']=inputs
report['script_SHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
dest=OUT/'target-score-temporal-accounting-v1.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
