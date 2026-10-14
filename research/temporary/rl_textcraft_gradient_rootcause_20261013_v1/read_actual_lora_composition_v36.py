"""Account actual LoRA BA drift before attributing nonlinear gradient growth to it.

CPU algebra on saved native tensors only, with pinned rank8/alpha16. Compute
the finite matrix identity Delta(BA)=DeltaB*A5+B5*DeltaA+DeltaB*DeltaA.
This is weight accounting, not a model, loss, optimizer, or derivative replica.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
labels=('delta_B_times_A5','B5_times_delta_A','delta_B_times_delta_A')

def factors(method,step):
    base=ROOT if method=='dt' and step==80 else SHARED
    directory=base/f'runs/textcraft-{method}-formal-native-graph-fresh-20261012-v1/textcraft-{method}/checkpoints/global_step_{step}/actor'
    models=[torch.load(directory/f'model_world_size_2_rank_{r}.pt',map_location='cpu',weights_only=False,mmap=True) for r in (0,1)]
    result={}
    for name in models[0]:
        if '.lora_' not in name:continue
        values=[m[name].to_local() if hasattr(m[name],'to_local') else m[name] for m in models]
        value=torch.cat(values,dim=0).double()
        assert list(value.shape)==list(models[0][name].shape)
        result[name]=value
    return result,str(directory)

def geometry(u,v):
    # Thin QR avoids materializing any dense4096x4096 update. Ordinary FP64
    # linear algebra on the literal low-rank factors, with rounding retained.
    _,ru=torch.linalg.qr(u,mode='reduced')
    _,rv=torch.linalg.qr(v.T,mode='reduced')
    core=ru@rv.T
    return dict(Frobenius_norm=float(torch.linalg.vector_norm(core)),
                operator_norm=float(torch.linalg.matrix_norm(core,ord=2)))

report=dict(unix=time.time(),scope=__doc__,algorithms={},scale=2.,GPU_model_calls=0,training_calls=0,GRPO_reruns=0,
            limits=['These are finite FP64 weight-matrix statistics, not gradient norms or independent causal shares.',
                    'A matrix cross term can scale quadratically without explaining the network VJP. Magnitude and time course must be compared to actual native data.',
                    'The Frobenius norm aggregated across different modules is not the network operator norm.',
                    'Same-step GRPO tensors are old saved controls, not a new GRPO model or training run.'])
for method,steps in [('dt',(40,80)),('grpo',(40,))]:
    initial,path5=factors(method,5)
    snapshots=[]
    for step in steps:
        current,path=factors(method,step)
        rows=[];gram=torch.zeros((3,3),dtype=torch.float64)
        effective_total=0.;cross_total=0.
        for name,a0 in initial.items():
            if '.lora_A.' not in name:continue
            bname=name.replace('.lora_A.','.lora_B.')
            b0=initial[bname];a=current[name];b=current[bname]
            da,db=a-a0,b-b0
            terms=[(2*db,a0),(2*b0,da),(2*db,da)]
            total=geometry(torch.cat((2*b,-2*b0),dim=1),torch.cat((a,a0),dim=0))
            term_stats={label:geometry(*pair) for label,pair in zip(labels,terms)}
            local_gram=torch.zeros((3,3),dtype=torch.float64)
            for i,(u,v) in enumerate(terms):
                for j,(w,z) in enumerate(terms):
                    local_gram[i,j]=((u.T@w)*(v@z.T)).sum()
            gram+=local_gram
            effective_total+=total['Frobenius_norm']**2
            cross_total+=term_stats[labels[2]]['Frobenius_norm']**2
            rows.append(dict(module=name.split('.lora_A.')[0],effective_displacement=total,
                             terms=term_stats,Gram=local_gram.tolist(),
                             matrix_identity_sumsq_residual=float(local_gram.sum())-total['Frobenius_norm']**2))
        snapshots.append(dict(step=step,reference_step=5,source=path,reference_source=path5,modules=rows,
                              global_module_collection_displacement_Frobenius=effective_total**.5,
                              global_cross_term_Frobenius=cross_total**.5,
                              cross_term_over_total_Frobenius=(cross_total/effective_total)**.5,
                              global_component_Gram=gram.tolist(),
                              matrix_identity_sumsq_residual=float(gram.sum())-effective_total))
        del current
    report['algorithms'][method]=snapshots
    del initial
assert not torch.cuda.is_initialized()
path=OUT/'actual-lora-composition-v36.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
