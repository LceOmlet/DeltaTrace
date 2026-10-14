"""Read direction changes of stored native adaptive updates, not noise labels.

Existing completed checkpoints only. No model/backward/update or substitute
optimizer is run. Numeric optimizer parameter IDs are matched only within the
same unchanged run, with their native group order and shard shapes checked.
"""
from collections import defaultdict
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
owner=Path(inspect.getsourcefile(inspect.unwrap(torch.optim.AdamW.step)))
report=dict(unix=time.time(),scope=__doc__,step_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    algorithms={},model_calls=0,optimizer_steps=0,GRPO_model_reruns=0,
    definition='Direction from already stored, bias-corrected native m/v. Same algebraic readout as v43; excludes decoupled decay and is not a bitwise optimizer replay.',
    limits=['Each direction is the just-completed final optimizer step at a saved checkpoint. A five-iteration gap contains20 native updates, not five.',
            'Endpoint direction cosine is not the average cosine or cancellation of all intervening updates.',
            'DT70-85 and GRPO60-65 are same-width late intervals of different runs/ages; do not call them matched-data interventions.',
            'Parameter IDs are matched only within each fixed original run; no shape-only model-to-optimizer name mapping is used.',
            'Update turning does not by itself prove that common return gradients caused internal gain.'])
for method,steps in [('dt',(70,75,80,85)),('grpo',(60,65))]:
    available={}
    for base in index['algorithms'][method]['bases']:
        for row in base['checkpoints']:
            if row['actor_state_files_present'] and int(base['completed_marker'])>=row['step']:
                available[row['step']]=row['path']
    previous=None;previous_ids=None;previous_step=None;output=[]
    for step in steps:
        current={};ids=[];group_configs=[];files=[]
        for rank in (0,1):
            path=Path(available[step])/f'optim_world_size_2_rank_{rank}.pt'
            files.append(str(path));opt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            for group_index,group in enumerate(opt['param_groups']):
                group_configs.append({k:v for k,v in group.items() if k!='params'})
                beta1,beta2=group['betas'];eps=group['eps']
                for pid in group['params']:
                    if pid not in opt['state']:continue
                    key=(rank,group_index,str(pid));ids.append(key)
                    s=opt['state'][pid];m=s['exp_avg'];v=s['exp_avg_sq'];shape=tuple(m.shape)
                    assert tuple(str(p) for p in m.placements)==('S(0)',)
                    side='A' if shape[0]==8 and shape[-1]!=8 else 'B' if shape[-1]==8 and shape[0]!=8 else 'other'
                    m=m.to_local().double();v=v.to_local().double();t=float(s['step']);assert t==4*step
                    direction=(m/(1-beta1**t))/(torch.sqrt(v/(1-beta2**t))+eps)
                    current[key]=dict(side=side,shape=shape,local_shape=tuple(direction.shape),direction=direction)
            del opt
        if previous_ids is not None:assert ids==previous_ids
        sums=defaultdict(float);old_sums=defaultdict(float);dots=defaultdict(float)
        for key,value in current.items():
            side=value['side'];d=value['direction']
            for label in ('all',side):sums[label]+=float(d.square().sum())
            if previous is not None:
                old=previous[key]
                assert (side,value['shape'],value['local_shape'])==(old['side'],old['shape'],old['local_shape'])
                for label in ('all',side):
                    old_sums[label]+=float(old['direction'].square().sum())
                    dots[label]+=float((d*old['direction']).sum())
        output.append(dict(iteration=step,files=files,groups=group_configs,
            state_count=len(current),stored_adaptive_direction_norm={k:v**.5 for k,v in sums.items()},
            previous_iteration=previous_step,
            endpoint_direction_cosine=None if previous is None else {k:dots[k]/(sums[k]*old_sums[k])**.5 for k in sums}))
        previous=current;previous_ids=ids;previous_step=step
    report['algorithms'][method]=output
assert not torch.cuda.is_initialized()
dest=OUT/'saved-update-turning-v44.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
