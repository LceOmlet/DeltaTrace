"""CPU-only inspection of captured native losses/gradients, split by real mask.

No new tolerance, clipping, correction or model inference is performed.
Absent inputs remain absent. A nonfinite tensor is located, not repaired.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

OUT=Path(os.environ['DT_ACTOR_INCIDENT_OUT'])
torch.set_num_threads(4)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(value):
    finite=torch.isfinite(value)
    selected=value[finite]
    return dict(shape=list(value.shape),dtype=str(value.dtype),count=value.numel(),
        nonfinite=int((~finite).sum()),min=float(selected.min()) if selected.numel() else None,
        max=float(selected.max()) if selected.numel() else None)


def inspect_rank(rank):
    path=OUT/f'rank{rank}-actual-update-input.pt'
    if not path.exists():return dict(rank=rank,status='actual_update_input_not_yet_captured')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    batch=payload['batch']
    source=json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
    sys.path.insert(0,source['verl_root'])
    from verl.trainer.ppo.core_algos import compute_policy_loss
    actor=OmegaConf.load(OUT/'effective-config.yaml').actor_rollout_ref.actor
    record=dict(rank=rank,input_path=str(path),input_sha256=sha(path),rows=batch['responses'].shape[0],
                original_meta_info=payload['meta_info'],steps=[])
    for step in range(4):
        path=OUT/f'rank{rank}-step{step}-before-native-step.pt'
        if not path.exists():continue
        saved=torch.load(path,map_location='cpu',weights_only=False)
        item=dict(index=step,path=str(path),sha256=sha(path),
                  raw_gradient_tensors=len(saved['raw_gradients']),microbatches=[])
        item['nonfinite_raw_gradients']=[dict(name=name,**stats(value)) for name,value in saved['raw_gradients'].items()
                                          if not bool(torch.isfinite(value).all())]
        for micro in saved['microbatches']:
            start=micro['index']*4;end=start+4
            length=batch['responses'].shape[-1]
            mask=batch['loss_mask'][start:end,-length:].bool()
            valid=batch['attention_mask'][start:end,-length:].bool()
            old=batch['old_log_probs'][start:end]
            ref=batch['ref_log_prob'][start:end]
            current=micro['log_prob']
            categories={}
            for name,selected in dict(action=mask,observation=valid&~mask,padding=~valid).items():
                values={key:stats(value[selected]) for key,value in micro.items()
                        if isinstance(value,torch.Tensor) and value.shape==mask.shape}
                difference=current[selected]-old[selected]
                ref_difference=ref[selected]-current[selected]
                # These are the owner's same exp operands, inspected on CPU.
                values.update(current_minus_old=stats(difference),ref_minus_current=stats(ref_difference),
                    PPO_exp_nonfinite=int((~torch.isfinite(torch.exp(difference))).sum()),
                    KL_exp_nonfinite=int((~torch.isfinite(torch.exp(ref_difference))).sum()))
                categories[name]=values
            computed=compute_policy_loss(old_log_prob=old,log_prob=current,
                advantages=batch['advantages'][start:end],response_mask=mask,
                cliprange=actor.clip_ratio,cliprange_low=actor.clip_ratio_low,
                cliprange_high=actor.clip_ratio_high,
                clip_ratio_c=actor.get('clip_ratio_c',3.0),loss_agg_mode=actor.loss_agg_mode)
            item['microbatches'].append(dict(index=micro['index'],categories=categories,
                native_policy_outputs=[float(x) for x in micro['policy_outputs']],
                official_CPU_readout=[float(x) for x in computed]))
        record['steps'].append(item)
    return record


result=dict(scope=__doc__,CUDA_initialized=torch.cuda.is_initialized(),
            ranks=[inspect_rank(rank) for rank in (0,1)])
print(json.dumps(result))
