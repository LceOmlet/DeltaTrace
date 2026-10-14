"""Prepared-only original Q/V/center coefficient diagnostic; no GPU submission.

This is a decomposition of the saved loss, not a proposed advantage estimator.
The original whitening owner supplies its actual mean and inverse standard
deviation through passive scalar readout. No whitening/PPO body is copied.
"""
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
from verl.utils import torch_functional as original_F

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-DT-QV-gradient-components-v31'
receipts=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
full=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
raw=full.batch['dt_token_advantages'].float()
mask=full.batch['response_mask'].float()
captured={'mean':[],'rsqrt':[]}
native_mean=original_F.masked_mean
native_rsqrt=torch.rsqrt
def read_mean(values,*args,**kwargs):
    result=native_mean(values,*args,**kwargs)
    if values.data_ptr()==raw.data_ptr():captured['mean'].append(result.detach().clone())
    return result
def read_rsqrt(value,*args,**kwargs):
    result=native_rsqrt(value,*args,**kwargs)
    captured['rsqrt'].append(result.detach().clone())
    return result
with patch.object(original_F,'masked_mean',read_mean),patch.object(torch,'rsqrt',read_rsqrt):
    recomputed=original_F.masked_whiten(raw,mask)*mask
assert torch.equal(recomputed,full.batch['advantages'])
assert captured['mean'] and len(captured['rsqrt'])==1
mean=captured['mean'][0];scale=captured['rsqrt'][0]
assert mean.ndim==scale.ndim==0
for value in captured['mean']:assert torch.equal(mean,value)
idx=torch.cat((torch.arange(32),torch.arange(128,160)))
data=full.select_idxs(idx);data.batch=data.batch.clone(recurse=False)
q=data.batch['dt_q_estimates'].float();v=data.batch['dt_v_estimates'].float();m=data.batch['response_mask'].float()
views=dict(full_DT=data.batch['advantages'].clone(),
    raw_Q=q*scale*m,minus_raw_V=-v*scale*m,
    original_center=(-mean*scale)*torch.ones_like(q)*m)
for name,value in views.items():data.batch['diagnostic_'+name]=value
residual=views['raw_Q']+views['minus_raw_V']+views['original_center']-views['full_DT']
launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
cfg=OmegaConf.load(Path(launch['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,value in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),value,force_add=True)
actor=cfg.actor_rollout_ref.actor
derivatives={}
for label,adv in views.items():
    c=torch.zeros_like(adv)
    for start in range(0,len(data),4):
        old=data.batch['old_log_probs'][start:start+4];lp=old.clone().requires_grad_()
        result=core_algos.compute_policy_loss(old_log_prob=old,log_prob=lp,advantages=adv[start:start+4],
            response_mask=m[start:start+4],cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
            cliprange_high=actor.clip_ratio_high,clip_ratio_c=actor.clip_ratio_c,loss_agg_mode=actor.loss_agg_mode)
        c[start:start+4]=torch.autograd.grad(result[0],lp)[0]/16
    derivatives[label]=c
coefficient_residual=derivatives['raw_Q']+derivatives['minus_raw_V']+derivatives['original_center']-derivatives['full_DT']
digests={}
for rank,r in enumerate(receipts):
    tensors=torch.load(r['lora_local'],map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for name,value in tensors.items():h.update(name.encode());h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    digests[str(rank)]=h.hexdigest()
DEST.mkdir(exist_ok=True)
path=DEST/'original81-QV-components.pkl';assert not path.exists();data.save_to_disk(str(path))
owner=Path(inspect.getsourcefile(original_F))
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),checkpoint=80,
    checkpoint_path=str(ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'),
    expected_trainable_SHA256_by_rank=digests,
    question='In the original DTPO policy gradient, is the large late norm carried by the factual Q term, the subtracted DT V term, or the original batch-centering term, and how do they cancel?',
    why_needed='Original transport and whole-batch whitening match PLAN. Native group partition localizes the large late PG to groups whose GRPO signal is zero, but does not identify which actual Q/V/center term creates it. Native last-decoder readout shows ordinary residual cancellation changes, not an evidenced operator defect. This decomposition returns to the DTPO-specific formula without estimating DT accuracy or changing credit quality.',
    diagnostic_advantage_views={name:'diagnostic_'+name for name in views},
    diagnostic_partition_kind='native_Q_V_center',
    cases=[dict(name='original81_QV',path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=64)],
    native_whitening_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    actual_native_whitening=dict(mean=float(mean),inverse_std=float(scale),recompute_exact=True),
    advantage_partition_max_abs_residual=float(residual.abs().max()),
    native_CPU_loss_coefficient_norms={name:float(c.double().square().sum().sqrt()) for name,c in derivatives.items()},
    native_CPU_loss_coefficient_partition_max_abs_residual=float(coefficient_residual.abs().max()),
    passive_boundary_readout_enabled=False,
    limits=['These are additive coefficients of the same saved loss at its native old-policy point, not new training advantages or a candidate method.',
        'Original full256 supplies native whitening. Original first global64 inputs, masks, identities and order remain unchanged.',
        'Coefficient and parameter gradients are distinct. CPU coefficient energy cannot be promoted to parameter-gradient contribution.',
        'Floating-point partition residuals are reported; no exact-sum claim or invented pass tolerance.',
        'Native PPO clipping is inactive at rho1. This partition does not describe every later clipped update or prove the85-iteration mechanism.',
        'No DT accuracy/causal value error is estimated. No optimizer/scheduler, new rollout, validation or GRPO model/training call.'],
    GPU_model_calls=0,parameter_backward=0,accepted_repair=None,deployed_repair=None)
assert not torch.cuda.is_initialized()
dest=DEST/'case-manifest.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
