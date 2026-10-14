"""Compare DTPO and original GRPO advantage coefficients on the saved same batch.

No model, rollout, optimizer, critic substitute, or loss implementation.
PPO measurements are read from its original running control separately.
"""
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
cfg=OmegaConf.load(Path(launch['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
actor=cfg.actor_rollout_ref.actor
core=Path(inspect.getsourcefile(core_algos))
assert hashlib.sha256(core.read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
receipts=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
views={}
for name in ('deltatrace','grpo'):
    view=data.select(deepcopy=True);view.batch=data.batch.clone(recurse=False)
    view=compute_advantage(view,adv_estimator=name,multi_turn=True,norm_adv_by_std_in_grpo=True)
    views[name]=view.batch['advantages']
assert torch.equal(views['deltatrace'],data.batch['advantages'])
mask=data.batch['response_mask'].bool()
targets=torch.zeros_like(mask)
for i,artifact in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
    positions=torch.as_tensor(artifact['retained_response_positions'])
    valid=positions>=0
    targets[i,valid]=torch.as_tensor(artifact['target_mask'],dtype=torch.bool)[positions[valid]]
assert not (targets&~mask).any()
rewards=data.batch['token_level_rewards'].sum(-1)
groups=defaultdict(list)
for i,uid in enumerate(data.non_tensor_batch['uid']):groups[str(uid)].append(i)
constant_positive=torch.zeros(len(data),dtype=torch.bool)
group_reports=[]
for uid,rows in groups.items():
    r=rewards[rows];constant=bool(torch.all(r==r[0]));positive=constant and float(r[0])>0
    if positive:constant_positive[rows]=True
    group_reports.append(dict(uid=uid,rows=rows,rewards=r.tolist(),same_reward=constant,
        same_positive_reward=positive,GRPO_max_abs=float(views['grpo'][rows][mask[rows]].abs().max()),
        DT_min=float(views['deltatrace'][rows][mask[rows]].min()),
        DT_max=float(views['deltatrace'][rows][mask[rows]].max())))
categories=dict(all_policy=mask,successful_target=targets&(rewards>0)[:,None],
    successful_source=mask&~targets&(rewards>0)[:,None],zero_reward=mask&(rewards==0)[:,None],
    same_positive_reward_group=mask&constant_positive[:,None],
    other_groups=mask&~constant_positive[:,None])
def statistics(values,selected):
    x=values[selected].double()
    return dict(tokens=x.numel(),min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,
        mean=float(x.mean()) if len(x) else None,L1=float(x.abs().sum()),L2=float(x.square().sum().sqrt()),
        positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
results={}
for name,a in views.items():
    c=torch.zeros_like(a)
    for start in range(0,len(data),4):
        old=data.batch['old_log_probs'][start:start+4]
        lp=old.detach().clone().requires_grad_()
        loss=core_algos.compute_policy_loss(old_log_prob=old,log_prob=lp,
            advantages=a[start:start+4],response_mask=mask[start:start+4],
            cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
            cliprange_high=actor.clip_ratio_high,clip_ratio_c=actor.clip_ratio_c,
            loss_agg_mode=actor.loss_agg_mode)
        c[start:start+4]=torch.autograd.grad(loss[0],lp)[0]/16
    results[name]=dict(advantages={k:statistics(a,s) for k,s in categories.items()},
        native_old_policy_loss_derivatives={k:statistics(c,s) for k,s in categories.items()},
        four_minibatch_coefficient_L2=[float(c[torch.cat((torch.arange(j,j+32),torch.arange(128+j,160+j)))].double().square().sum().sqrt()) for j in (0,32,64,96)])
sizes=sorted(set(len(r) for r in groups.values()))
report=dict(unix=time.time(),scope=__doc__,rows=len(data),groups=len(groups),group_sizes=sizes,
    original_DT_advantage_recompute_exact=True,core_owner=dict(path=str(core),SHA256=hashlib.sha256(core.read_bytes()).hexdigest()),
    advantage_owner=dict(path=inspect.getsourcefile(compute_advantage),SHA256=hashlib.sha256(Path(inspect.getsourcefile(compute_advantage)).read_bytes()).hexdigest()),
    source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),group_reports=group_reports,results=results,
    structural_facts=['Original GRPO is exactly zero on a group whose sampled rewards are identical. Its group-local sample standard deviation bounds each nonzero coefficient by(n-1)/sqrt(n), before the positive epsilon.',
        'DTPO uses original global token whitening of Q-V. That unit-variance preprocessing does not imply group-local zero means or a small per-token bound independent of the full token count.',
        'PPO/GAE must be compared with its actual critic outputs. No ideal or substitute critic is inserted into this saved DT batch.'],
    limits=['These are actual original-loss derivatives with respect to sampled log probabilities, not parameter gradients or causal shares.',
        'All four minibatches are evaluated at their saved old-policy point; intervening original optimizer updates are not replayed.',
        'Group-wise centering and coefficient bounds alone do not prove a bounded parameter norm, a unique root cause or a repair.',
        'The existing same-forward DT/GRPO parameter-gradient comparison supplies the measured magnitude/direction contrast; no GRPO model rerun is made.'],
    model_calls=0,parameter_backward=0,optimizer=0,DT_calls=0,rollout=0,GRPO_training_reruns=0,
    rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
assert not torch.cuda.is_initialized()
dest=OUT/'native-advantage-stability-contrast-v22.json'
assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
