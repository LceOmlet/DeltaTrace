"""Read actual original-loss derivatives of the two prepared factual carriers."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/matched-action-observation-target-derivatives-v20'
manifest=json.loads((DEST/'case-manifest.json').read_bytes())
launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
cfg=OmegaConf.load(Path(launch['effective_environment']['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
actor=cfg.actor_rollout_ref.actor
owner=Path(inspect.getsourcefile(core_algos))
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
outputs=[];target_ids=[];counts=[];derivatives=[]
for case in manifest['cases']:
    data=DataProto.load_from_disk(case['path']);mask=data.batch['response_mask']
    derivative=torch.zeros_like(data.batch['old_log_probs'])
    for start in range(0,len(data),4):
        old=data.batch['old_log_probs'][start:start+4];lp=old.clone().requires_grad_()
        loss=core_algos.compute_policy_loss(old_log_prob=old,log_prob=lp,
            advantages=data.batch['advantages'][start:start+4],response_mask=mask[start:start+4],
            cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
            cliprange_high=actor.clip_ratio_high,clip_ratio_c=actor.clip_ratio_c,
            loss_agg_mode=actor.loss_agg_mode)
        derivative[start:start+4]=torch.autograd.grad(loss[0],lp)[0]/16
    target=data.batch['diagnostic_target_mask'].bool()
    n=len(case['coefficients'])
    outputs.append([derivative[i,target[i]] for i in range(n)])
    target_ids.append([data.batch['responses'][i,target[i]] for i in range(n)])
    derivatives.append(float(derivative.double().square().sum().sqrt()))
    counts.append(dict(name=case['name'],rows=len(data),active_target_derivative_rows=n,
        non_target_derivative_max_abs=float(derivative[~target].abs().max()),
        zero_derivative_filler_max_abs=float(derivative[n:].abs().max())))
assert len(outputs)==2 and all(torch.equal(a,b) for a,b in zip(*target_ids))
differences=[float((a-b).abs().max()) for a,b in zip(*outputs)]
report=dict(unix=time.time(),scope=__doc__,cases=counts,
    target_ID_sequences_exact_equal=True,per_pair_target_derivative_max_abs_differences=differences,
    coefficient_vector_L2_by_case=derivatives,
    native_core_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    model_calls=0,parameter_backward=0,optimizer=0,DT=0,rollout=0,
    scope_limit='Original CPU loss derivatives at log_prob=old_log_prob, with the unchanged native accumulation/rank mean. This checks coefficient comparability only; it does not predict parameter-gradient norms or accept a repair.')
assert not torch.cuda.is_initialized()
dest=DEST/'original-loss-coefficient-comparability.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
