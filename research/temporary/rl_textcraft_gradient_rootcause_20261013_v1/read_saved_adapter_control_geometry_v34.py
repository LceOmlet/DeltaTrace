"""Compare existing matched-step DTPO/GRPO parameter drift on CPU only.

No GRPO model/forward/backward/training is rerun. Actual saved parameter
displacement tests whether a generic larger LoRA scale explains the measured
internal propagation gain; it does not replace directional derivative data.
"""
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
index=json.loads((OUT/'saved-control-checkpoint-index-v34.json').read_bytes())
report=dict(unix=time.time(),scope=__doc__,algorithms={},GPU_model_calls=0,optimizer_updates=0,GRPO_reruns=0,
            limits=['Parameter norms and displacement are not gradient norms or causal contributions.',
                    'Matched completed steps5/40 only; no missing checkpoint or intermediate optimizer path is fabricated.',
                    'The two algorithms have different sampled inputs and update directions. This checks a generic scale explanation, not a controlled algorithm intervention.'])
for method in ('dt','grpo'):
    base=index['algorithms'][method]['bases'][1]
    previous={};snapshots=[]
    for step in (5,40):
        item=next(c for c in base['checkpoints'] if c['step']==step)
        assert item['actor_state_files_present'] and int(base['completed_marker'])>=step
        paths=[Path(item['path'])/f'model_world_size_2_rank_{rank}.pt' for rank in (0,1)]
        models=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
        rows=[];sums={'A':0.,'B':0.};delta_sums={'A':0.,'B':0.};digest=hashlib.sha256()
        for name in models[0]:
            if '.lora_' not in name:continue
            local=[m[name].to_local() if hasattr(m[name],'to_local') else m[name] for m in models]
            value=torch.cat(local,dim=0).float()
            assert list(value.shape)==list(models[0][name].shape)
            side='A' if '.lora_A.' in name else 'B'
            sq=float(value.double().square().sum());sums[side]+=sq
            row=dict(name=name,shape=list(value.shape),Frobenius_norm=sq**.5,
                     operator_norm=float(torch.linalg.matrix_norm(value,ord=2)))
            digest.update(name.encode());digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            if step==5:previous[name]=value.clone()
            else:
                difference=value-previous[name]
                ds=float(difference.double().square().sum());delta_sums[side]+=ds
                row.update(displacement_Frobenius_norm=ds**.5,
                           displacement_operator_norm=float(torch.linalg.matrix_norm(difference,ord=2)))
            rows.append(row)
        snapshots.append(dict(step=step,paths=[str(p) for p in paths],parameters=rows,
                              global_parameter_norms={k:v**.5 for k,v in sums.items()},
                              global_displacement_from_step5_norms=None if step==5 else {k:v**.5 for k,v in delta_sums.items()},
                              reconstructed_LoRA_FP32_SHA256=digest.hexdigest()))
        del models
    report['algorithms'][method]=snapshots
report['matched_step_DT_over_GRPO_parameter_norm_ratios']={str(step):{
    side:report['algorithms']['dt'][i]['global_parameter_norms'][side]/report['algorithms']['grpo'][i]['global_parameter_norms'][side]
    for side in ('A','B')} for i,step in enumerate((5,40))}
assert not torch.cuda.is_initialized()
path=OUT/'saved-adapter-control-geometry-v34.json'
assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
