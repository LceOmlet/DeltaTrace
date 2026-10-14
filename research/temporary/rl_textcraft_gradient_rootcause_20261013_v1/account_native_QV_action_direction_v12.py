"""Read finite Action-logit directions through the original fixed VERL loss.

Only saved iteration81 inputs/probabilities and native CPU scalar derivatives
are used. There is no model forward, parameter backward, optimizer update or
replacement PPO/whitening implementation.
"""
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils import torch_functional as native_F

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
cfg=OmegaConf.load(Path(launch['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
actor=cfg.actor_rollout_ref.actor
core_path=Path(inspect.getsourcefile(core_algos))
assert hashlib.sha256(core_path.read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
mask=data.batch['response_mask']
raw=data.batch['dt_token_advantages']
scales=[];original_rsqrt=torch.rsqrt
def read_scale(*args,**kwargs):
    result=original_rsqrt(*args,**kwargs);scales.append(result.detach().clone());return result
with patch.object(torch,'rsqrt',read_scale):
    recomputed=native_F.masked_whiten(raw,mask,shift_mean=True)*mask
assert torch.equal(recomputed,data.batch['advantages']) and len(scales)==1
scale=scales[0];mean=native_F.masked_mean(raw,mask)
components={'actual_DT':data.batch['advantages'],
            'Q':data.batch['dt_q_estimates']*scale*mask,
            'negative_V':-data.batch['dt_v_estimates']*scale*mask,
            'centering':torch.ones_like(raw)*(-mean*scale)*mask}
reconstruction=components['Q']+components['negative_V']+components['centering']
forks=json.loads((OUT/'same-prefix-DT-values-v5.json').read_bytes())['forks']
groups=[];abstentions=[]
for fork in forks:
    action=[b for b in fork['branches'] if b['target'] and b['token']=='Action']
    if len(action)!=1:continue
    action_id=action[0]['token_id'];by_B4=defaultdict(list)
    for branch in fork['branches']:
        for row in branch['occurrences']:
            by_B4[row['row']//4].append(row)
    for B4,rows in by_B4.items():
        chosen=[r for r in rows if r['token_id']==action_id]
        if not chosen or len({r['token_id'] for r in rows})<2:continue
        if len({r['native_old_logp'] for r in chosen})!=1:
            abstentions.append({'prefix':fork['prefix_SHA256'],'B4':B4,'reason':'Saved Action probabilities differ within this B4.'});continue
        signatures=[]
        width=data.batch['responses'].shape[1];prompt=data.batch['input_ids'].shape[1]-width
        for r in rows:
            i,j=r['row'],r['position'];active=data.batch['attention_mask'][i,:prompt+j].bool()
            ids=data.batch['input_ids'][i,:prompt+j][active]
            positions=data.batch['position_ids'][i,...,:prompt+j][...,active]
            signatures.append((ids.numpy().tobytes(),positions.numpy().tobytes()))
        if len(set(signatures))!=1:
            abstentions.append({'prefix':fork['prefix_SHA256'],'B4':B4,'reason':'Saved active IDs or positions do not match.'});continue
        probability=float(torch.tensor(chosen[0]['native_old_logp'],dtype=torch.float64).exp())
        groups.append({'prefix_SHA256':fork['prefix_SHA256'],'prefix_tail':fork['prefix_tail'],
            'B4':B4,'rows':rows,'Action_token_id':action_id,'native_Action_probability':probability})

gradients={};loss_records={}
native_accumulation=(int(actor.ppo_mini_batch_size)//2)//int(actor.ppo_micro_batch_size_per_gpu)
native_global_scale=1/(native_accumulation*2)
for B4 in sorted({g['B4'] for g in groups}):
    start=B4*4;old=data.batch['old_log_probs'][start:start+4]
    for name,coefficients in components.items():
        lp=old.detach().clone().requires_grad_()
        result=core_algos.compute_policy_loss(old_log_prob=old,log_prob=lp,
            advantages=coefficients[start:start+4],response_mask=mask[start:start+4],
            cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
            cliprange_high=actor.clip_ratio_high,clip_ratio_c=actor.clip_ratio_c,
            loss_agg_mode=actor.loss_agg_mode)
        gradient,=torch.autograd.grad(result[0],lp)
        gradients[B4,name]=gradient.detach().double()*native_global_scale
        loss_records[B4,name]={'loss':float(result[0]),'clipfrac':float(result[1]),
                               'ppo_kl':float(result[2]),'clipfrac_lower':float(result[3])}
for g in groups:
    g['Action_logit_descent_direction']={}
    for name in components:
        value=0.0
        for r in g['rows']:
            derivative=(1.0 if r['token_id']==g['Action_token_id'] else 0.0)-g['native_Action_probability']
            value-=float(gradients[g['B4'],name][r['row']%4,r['position']])*derivative
        g['Action_logit_descent_direction'][name]=value
    v=g['Action_logit_descent_direction']
    g['Q_plus_centering_descent']=v['Q']+v['centering']
    g['sum_Q_negativeV_centering_descent']=v['Q']+v['negative_V']+v['centering']
    g['baseline_term_favors_Action']=v['negative_V']>0
    g['baseline_term_changes_descent_from_negative_to_positive']=g['Q_plus_centering_descent']<0<v['actual_DT']
    g['original_native_loss_records']={k:loss_records[g['B4'],k] for k in components}
report={'unix':time.time(),'scope':__doc__,'original_input_sources':sources,
    'core_owner':{'path':str(core_path),'SHA256':hashlib.sha256(core_path.read_bytes()).hexdigest()},
    'whitening_owner':{'path':inspect.getsourcefile(native_F.masked_whiten),
                       'original_recompute_exact':True,'observed_mean':float(mean),'observed_scale':float(scale)},
    'coefficient_reconstruction_max_abs':float((reconstruction-components['actual_DT']).abs().max()),
    'CPU_original_loss_derivative_evaluations':len(gradients),'new_model_calls':0,
    'new_parameter_backward_optimizer_scheduler_calls':0,'copied_training_function_bodies':0,
    'native_accumulation':native_accumulation,'native_cross_rank_mean_size':2,
    'groups':groups,'abstentions':abstentions,
    'limits':[
        'These are finite saved-batch old-policy-point logit directions, not conditional population Q/V estimates, a proof of estimator bias, or a full parameter update prediction.',
        'The softmax chain rule uses the saved within-B4 Action probability and matching active prefixes/positions; native floating-point logit backward rounding is not reconstructed.',
        'All four global minibatches are evaluated at their saved old-policy point; original intervening optimizer updates are not replayed.',
        'The Q/V/centering split has the reported floating-point reconstruction residual. No new whitening formula or PPO objective is implemented.',
        'A component direction can oppose or exceed the total; contributions are not independent causal percentages.',
        'This explains a local objective preference, not yet the entire85-iteration norm trend or a validated repair.']}
assert not torch.cuda.is_initialized()
dest=OUT/'native-QV-Action-direction-v12.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
