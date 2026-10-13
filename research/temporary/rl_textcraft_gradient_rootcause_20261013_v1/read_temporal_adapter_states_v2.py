"""CPU-only exact LoRA/Adam tensor statistics from existing native checkpoints.

Memory-map native files and read trainable adapter/state tensors only. No model
construction, checkpoint save/restore, evaluation, or optimizer update.
"""
import json
from pathlib import Path
import time
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
REL=Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
steps=[5,10,15,20,25,40,45,80,85]
report={'scope':__doc__,'unix':time.time(),'GPU_used':False,'checkpoints':[]}
def local(x):
    return x.to_local() if hasattr(x,'to_local') else x
for step in steps:
    base=(ROOT if step>=80 else SHARED)/REL/f'global_step_{step}/actor'
    row={'step':step,'source_root':str(base),'ranks':[]}
    for rank in (0,1):
        path=base/f'model_world_size_2_rank_{rank}.pt'
        model=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        names=[k for k in model if 'lora_' in k]
        values={k:local(model[k]).double() for k in names}
        assert len(names)==496
        kinds={}
        for k,x in values.items():
            label='A' if '.lora_A.' in k else 'B'
            g=kinds.setdefault(label,{'sumsq':0.,'elements':0,'max_abs':0.})
            g['sumsq']+=x.square().sum().item();g['elements']+=x.numel();g['max_abs']=max(g['max_abs'],x.abs().max().item())
        opath=base/f'optim_world_size_2_rank_{rank}.pt'
        optim=torch.load(opath,map_location='cpu',weights_only=False,mmap=True)
        states=optim['state']
        all_params=[p for g in optim['param_groups'] for p in g['params']]
        params=[p for p in all_params if p in states]
        rr={'rank':rank,'files':{'model':str(path),'optimizer':str(opath)},'LoRA_tensor_names':names,
            'adapter_group_statistics':kinds,'optimizer_state_count':len(states),'all_optimizer_parameter_count':len(all_params),'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in optim['param_groups']],
            'state_shapes':{},'moments_by_module':{}}
        assert len(params)==len(names)
        for k,pid in zip(names,params):
            state=states[pid];x=values[k];m=local(state['exp_avg']).double();v=local(state['exp_avg_sq']).double()
            assert m.shape==x.shape and v.shape==x.shape,(k,m.shape,x.shape)
            rr['state_shapes'][k]=list(x.shape)
            rr['moments_by_module'][k]={'weight_sumsq':x.square().sum().item(),'exp_avg_sumsq':m.square().sum().item(),
                'exp_avg_sq_sum':v.sum().item(),'optimizer_step':float(state['step'])}
        rr['parameter_order_scope']='LoRA state_dict order matched optimizer local tensor shapes; not a replacement for native optimizer ownership'
        row['ranks'].append(rr)
        del model,optim,values
    report['checkpoints'].append(row)
path=OUT/'temporal-adapter-states-v2.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
