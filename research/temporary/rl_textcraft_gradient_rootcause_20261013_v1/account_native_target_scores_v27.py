"""Account native target score-force bounds on the unchanged matched histories.

The loss derivative comes from original VERL CPU autograd, not a copied PPO.
The softmax bound is algebra on actual owner probabilities; it is not an
alternative head implementation, a measured full-logit norm, or a repair test.
"""
from collections import defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/(sys.argv[1] if len(sys.argv)>1 else 'matched-target-score-readout-v27')
assert DEST.parent==OUT
manifest=json.loads((DEST/'case-manifest.json').read_bytes())
state=json.loads((DEST/'result.json').read_bytes())
assert state['phase'] in ('complete_native_target_score_readout_not_repair','complete_native_gradient_chain_localization_not_repair')
assert all(state[k]==0 for k in ('optimizer_updates','scheduler_updates','DT_calls','rollout_calls','validation_calls','GRPO_calls'))
launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
cfg=OmegaConf.load(Path(launch['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
actor=cfg.actor_rollout_ref.actor
owner=Path(inspect.getsourcefile(core_algos))
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
report=dict(unix=time.time(),scope=__doc__,cases=[],script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_core_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    bound_definition='For temperature-scaled vocabulary logits, z_i=c_i(one_hot(y_i)-p_i). With a normalized categorical distribution, |c_i|(1-p_i[y_i]) <= ||z_i||2 <= sqrt(2)|c_i|(1-p_i[y_i]). Squared bounds add over token positions. Native rounded probabilities are retained; vocabulary vectors are not reconstructed.',
    limit='These are output-force magnitude bounds, not parameter gradients or independent causal shares. No training outcome or full historical growth claim.',
    GPU_model_calls=0,parameter_backward=0,optimizer_updates=0,DT_calls=0,GRPO_calls=0,rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
coefficients=[];target_ids=[]
for case,observation in zip(manifest['cases'],state['observations']):
    assert case['name']==observation['iteration']
    path=Path(observation['native_probability_carrier'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==observation['SHA256']
    data=DataProto.load_from_disk(str(path))
    mask=data.batch['response_mask']
    old=data.batch['old_log_probs']
    derivative=torch.zeros_like(old)
    for start in range(0,len(data),4):
        lp=old[start:start+4].clone().requires_grad_()
        result=core_algos.compute_policy_loss(old_log_prob=old[start:start+4],log_prob=lp,
            advantages=data.batch['diagnostic_target_legacy'][start:start+4],response_mask=mask[start:start+4],
            cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
            cliprange_high=actor.clip_ratio_high,clip_ratio_c=actor.clip_ratio_c,
            loss_agg_mode=actor.loss_agg_mode)
        derivative[start:start+4]=torch.autograd.grad(result[0],lp)[0]/16
    selected=derivative!=0
    assert not bool((selected&~data.batch['diagnostic_target_mask'].bool()).any())
    ids=data.batch['responses']
    lp=old.double()
    deficit=-torch.expm1(lp)
    force=derivative.double()*deficit
    total_lower_squared=float(force[selected].square().sum())
    pairs=[]
    for i in range(15):
        active=selected[i]
        values=lp[i,active]
        pairs.append(dict(row=i,target_tokens=int(active.sum()),
            native_target_NLL=float(-values.sum()),
            target_probability_quantiles=torch.quantile(values.exp(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
            coefficient_L2=float(derivative[i,active].double().square().sum().sqrt()),
            score_lower_squared=float(force[i,active].square().sum())))
    by_id=defaultdict(lambda:dict(tokens=0,score_lower_squared=0.,coefficient_squared=0.,NLL=0.))
    for token in torch.unique(ids[selected]).tolist():
        active=selected&(ids==token)
        v=by_id[str(token)]
        v.update(tokens=int(active.sum()),score_lower_squared=float(force[active].square().sum()),
            coefficient_squared=float(derivative[active].double().square().sum()),NLL=float(-lp[active].sum()))
    report['cases'].append(dict(case=case['name'],native_probability_carrier=str(path),
        temperature=data.meta_info.get('temperature'),
        target_tokens=int(selected.sum()),
        coefficient_vector_L2=float(derivative.double().square().sum().sqrt()),
        native_target_logp_gt_zero=int((lp[selected]>0).sum()),
        score_norm_bounds=[math.sqrt(total_lower_squared),math.sqrt(2*total_lower_squared)],
        target_NLL=float(-lp[selected].sum()),pairs=pairs,score_by_token_ID=dict(by_id)))
    coefficients.append([derivative[i,selected[i]] for i in range(15)])
    target_ids.append([ids[i,selected[i]] for i in range(15)])
assert all(torch.equal(a,b) for a,b in zip(*target_ids))
report['per_pair_coefficient_max_abs_difference']=[float((a-b).abs().max()) for a,b in zip(*coefficients)]
a,b=[c['score_norm_bounds'] for c in report['cases']]
report['late_over_early_score_norm_bounds']=[b[0]/a[1],b[1]/a[0]]
assert not torch.cuda.is_initialized()
path=DEST/'native-target-score-accounting.json'
assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
