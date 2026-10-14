"""Read actual saved LoRA parameter displacement through the abnormal interval.

CPU checkpoint accounting, no optimizer formula or training implementation.
Use completed historical controls, never a GRPO model rerun.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
index=json.loads((OUT/'saved-control-checkpoint-index-v34.json').read_bytes())
report=dict(unix=time.time(),scope=__doc__,algorithms={},GPU_model_calls=0,optimizer_updates=0,GRPO_reruns=0,
            limits=['This is the norm of net saved parameter displacement divided by interval width, not the mean of individual optimizer-step norms.',
                    'Intervals of unequal widths must not be treated as equal-duration samples; consecutive five-iteration intervals are marked explicitly.',
                    'Parameter velocity is not gradient magnitude or a causal share of internal VJP amplification.',
                    'Models and optimizer states are not stepped, reconstructed, or simulated.'])
for method in ('dt','grpo'):
    available={}
    for base in index['algorithms'][method]['bases']:
        for row in base['checkpoints']:
            if row['actor_state_files_present'] and row['step']<= (85 if method=='dt' else 65):
                available[row['step']]=row['path']
    previous=None;previous_step=None;previous_delta=None;rows=[]
    for step,path in sorted(available.items()):
        models=[torch.load(Path(path)/f'model_world_size_2_rank_{r}.pt',map_location='cpu',weights_only=False,mmap=True) for r in (0,1)]
        values={}
        for name in models[0]:
            if '.lora_' not in name:continue
            v=torch.cat([m[name].to_local() if hasattr(m[name],'to_local') else m[name] for m in models],dim=0).float()
            assert list(v.shape)==list(models[0][name].shape)
            values[name]=v
        if previous is not None:
            width=step-previous_step
            sums=defaultdict(float);old_sums=defaultdict(float);dots=defaultdict(float);family_sums=defaultdict(float);delta={}
            for name,v in values.items():
                d=v-previous[name];delta[name]=d
                side='A' if '.lora_A.' in name else 'B';sq=float(d.double().square().sum());sums[side]+=sq
                family=name.split('layers.')[1].split('.lora_')[0].split('.',1)[1]
                family_sums[family]+=sq
                if previous_delta is not None:
                    old=previous_delta[name]
                    old_sums[side]+=float(old.double().square().sum())
                    dots[side]+=float((d.double()*old.double()).sum())
            rows.append(dict(from_step=previous_step,to_step=step,width=width,same_five_iteration_width=width==5,
                             source=str(path),net_displacement_norm={k:v**.5 for k,v in sums.items()},
                             net_displacement_norm_per_iteration={k:v**.5/width for k,v in sums.items()},
                             cosine_with_previous_saved_interval=None if previous_delta is None else {
                                 k:dots[k]/(sums[k]*old_sums[k])**.5 for k in sums},
                             family_net_displacement_norm={k:v**.5 for k,v in family_sums.items()}))
            previous_delta=delta
        previous=values;previous_step=step
        del models
    report['algorithms'][method]=rows
assert not torch.cuda.is_initialized()
path=OUT/'saved-parameter-velocity-v37.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
