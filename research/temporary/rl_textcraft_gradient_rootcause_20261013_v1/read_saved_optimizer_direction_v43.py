"""Read the stored native AdamW direction through the gradient-growth interval.

This is FP64 statistical accounting of existing m/v tensors, not an optimizer
implementation: no state transition, gradient synthesis, optimizer.step or
parameter update is executed. Native beta/epsilon/lr values are read unchanged.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
index=json.loads((OUT/'saved-control-checkpoint-index-v34.json').read_bytes())
owner=Path(inspect.getsourcefile(torch.optim.AdamW))
report=dict(unix=time.time(),scope=__doc__,native_optimizer_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    definition='Stored bias-corrected moment direction d_t=(m_t/(1-beta1**t))/(sqrt(v_t/(1-beta2**t))+epsilon). Its FP64 norm accounts the adaptive part of the just-completed native AdamW step. It is not a simulated next step.',
    algorithms={},model_calls=0,optimizer_steps=0,GRPO_model_reruns=0,
    limits=['No actual past gradient is reconstructed, and no optimizer state or parameter is changed.',
            'The readout uses FP64 algebra of saved native FP32 moments; it is not a bitwise replay of native foreach/fused rounding.',
            'The adaptive direction excludes decoupled weight decay. Its lr-scaled norm is not the norm of an entire five-iteration displacement.',
            'The same optimizer equation is not a causal attribution to a particular loss component.',
            'DT and GRPO are compared at matched steps5/40 only; GRPO65 is a later existing control, not a fabricated GRPO80 checkpoint.'])
for method,steps in [('dt',(5,40,80,85)),('grpo',(5,40,65))]:
    available={}
    for base in index['algorithms'][method]['bases']:
        for row in base['checkpoints']:
            if row['actor_state_files_present'] and int(base['completed_marker'])>=row['step']:
                available[row['step']]=row['path']
    output=[]
    for step in steps:
        base=Path(available[step]);row=dict(iteration=step,files=[],groups=[],state_steps=[],placements=[],
            m_sumsq=0.,v_sum=0.,direction_sumsq=0.,lr_scaled_direction_sumsq=0.,parameter_state_count=0,elements=0)
        counters=set();placements=set()
        for rank in (0,1):
            path=base/f'optim_world_size_2_rank_{rank}.pt'
            opt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            row['files'].append(dict(path=str(path),bytes=path.stat().st_size))
            for group in opt['param_groups']:
                cfg={k:v for k,v in group.items() if k!='params'}
                if cfg not in row['groups']:row['groups'].append(cfg)
                beta1,beta2=group['betas'];epsilon=group['eps'];lr=group['lr']
                assert not group['amsgrad'] and not group['maximize']
                for parameter in group['params']:
                    if parameter not in opt['state']:continue
                    state=opt['state'][parameter]
                    m=state['exp_avg'];v=state['exp_avg_sq']
                    placement=tuple(str(p) for p in m.placements) if hasattr(m,'placements') else ('plain_local_saved',)
                    placements.add(placement)
                    # The native per-rank checkpoints store S(0) shards.
                    if hasattr(m,'placements'):assert placement==('S(0)',)
                    m=m.to_local() if hasattr(m,'to_local') else m
                    v=v.to_local() if hasattr(v,'to_local') else v
                    m=m.double();v=v.double()
                    t=float(state['step']);counters.add(t)
                    assert t==4*step,(method,step,t)
                    assert torch.isfinite(m).all() and torch.isfinite(v).all() and (v>=0).all()
                    direction=(m/(1-beta1**t))/(torch.sqrt(v/(1-beta2**t))+epsilon)
                    ss=float(direction.square().sum())
                    row['m_sumsq']+=float(m.square().sum());row['v_sum']+=float(v.sum())
                    row['direction_sumsq']+=ss;row['lr_scaled_direction_sumsq']+=lr*lr*ss
                    row['parameter_state_count']+=1;row['elements']+=m.numel()
            del opt
        row.update(state_steps=sorted(counters),placements=[list(p) for p in sorted(placements)],
                   m_norm=row['m_sumsq']**.5,root_sum_v=row['v_sum']**.5,
                   stored_adaptive_direction_norm=row['direction_sumsq']**.5,
                   lr_scaled_adaptive_part_norm=row['lr_scaled_direction_sumsq']**.5)
        output.append(row)
        (OUT/'saved-optimizer-direction-progress-v43.json').write_text(json.dumps(dict(method=method,step=step,unix=time.time()))+'\n')
    report['algorithms'][method]=output
assert not torch.cuda.is_initialized()
dest=OUT/'saved-optimizer-direction-v43.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
