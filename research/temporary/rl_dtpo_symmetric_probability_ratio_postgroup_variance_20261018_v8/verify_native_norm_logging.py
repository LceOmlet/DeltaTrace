"""Verify the actual PyTorch norm return/full_tensor on the same gradient, no optimizer/model."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

dist.init_process_group('gloo')
world=dist.get_world_size();rank=dist.get_rank()
mesh=init_device_mesh('cpu',(world,))
# Integer FP64 entries have an exactly representable sum of squares; lengths
# divide both 2 and 8. Compare official clip_grad_norm_ returns, not a new norm.
gradient=torch.arange(1,65,dtype=torch.float64)
single=torch.nn.Parameter(torch.zeros_like(gradient));single.grad=gradient.clone()
reference=torch.nn.utils.clip_grad_norm_([single],float('inf'),foreach=False)
parameter=torch.nn.Parameter(distribute_tensor(torch.zeros_like(gradient),mesh,[Shard(0)]))
parameter.grad=distribute_tensor(gradient.clone(),mesh,[Shard(0)])
native=torch.nn.utils.clip_grad_norm_([parameter],float('inf'),foreach=False)
local=native.detach().item();global_value=native.full_tensor()
torch.testing.assert_close(global_value,reference,rtol=0,atol=0)
assert torch.equal(parameter.grad.full_tensor(),gradient)
assert not torch.cuda.is_initialized()
rows=[None]*world;dist.all_gather_object(rows,dict(rank=rank,local=local,global_value=global_value.item()))
if rank==0:
    out=Path(os.environ['DT_V8_NORM_AUDIT'])/f'official-norm-{world}rank.json'
    out.write_text(json.dumps(dict(status='passed',world_size=world,dtype='float64',
        original_api='torch.nn.utils.clip_grad_norm_',materialization='DTensor.full_tensor',
        global_reference=reference.item(),ranks=rows,model_calls=0,optimizer_steps=0,
        exact_comparison_reason='identical integer FP64 gradient, exactly representable squared sum',
        GPU_initialized=False),indent=2)+'\n')
dist.destroy_process_group()
