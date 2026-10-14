"""Read actual checkpoint LoRA geometry after the measured G/H gain changed.

This is parameter accounting only. Parameter norms do not replace gradient
norms, and small operator-norm changes do not rule out directional changes.
"""
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
REL=Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
previous={};report=dict(unix=time.time(),scope=__doc__,checkpoints=[],GPU_model_calls=0,optimizer_updates=0,
    limits=['LoRA operator/Frobenius norms are scalar parameter statistics, not directional gradient gains or causal contributions.',
        'This checks whether a uniform change in the existing trainable projection scale is sufficient; it cannot by itself exclude changes in activation direction, propagation or cancellation.'])
for step,base in [(5,SHARED),(80,ROOT)]:
    paths=[base/REL/f'global_step_{step}/actor/model_world_size_2_rank_{rank}.pt' for rank in (0,1)]
    models=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
    rows=[]
    for name in models[0]:
        if '.lora_' not in name:continue
        values=[m[name].to_local() if hasattr(m[name],'to_local') else m[name] for m in models]
        value=torch.cat(values,dim=0).float()
        assert list(value.shape)==list(models[0][name].shape)
        row=dict(name=name,shape=list(value.shape),Frobenius_norm=float(torch.linalg.vector_norm(value)),
            operator_norm=float(torch.linalg.matrix_norm(value,ord=2)),max_abs=float(value.abs().max()))
        if step==5:previous[name]=(value.clone(),row)
        else:
            old,old_row=previous[name]
            row.update(Frobenius_norm_ratio=row['Frobenius_norm']/old_row['Frobenius_norm'] if old_row['Frobenius_norm'] else None,
                operator_norm_ratio=row['operator_norm']/old_row['operator_norm'] if old_row['operator_norm'] else None,
                delta_Frobenius_norm=float(torch.linalg.vector_norm(value-old)),
                delta_operator_norm=float(torch.linalg.matrix_norm(value-old,ord=2)))
        rows.append(row)
    report['checkpoints'].append(dict(checkpoint=step,paths=[str(p) for p in paths],parameters=rows))
    del models
assert not torch.cuda.is_initialized()
dest=OUT/'native-adapter-geometry-v29.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
