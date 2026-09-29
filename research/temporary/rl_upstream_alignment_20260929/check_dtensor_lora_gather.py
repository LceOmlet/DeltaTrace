"""Native DTensor gather at the actual rank-one LoRA parameter shapes."""
import json
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group(backend='cpu:gloo,cuda:nccl')
mesh = init_device_mesh('cuda', (2,))
for device in ('cpu', 'cuda'):
    for shape in ((1, 4096), (4096, 1), (1, 2048), (2048, 1)):
        x = torch.arange(shape[0]*shape[1], device='cuda', dtype=torch.float32).reshape(shape).to(torch.bfloat16)
        value = distribute_tensor(x, mesh, [Shard(0)]).to(device)
        print(json.dumps(dict(rank=dist.get_rank(), device=device, shape=shape,
            local_shape=list(value.to_local().shape), phase='before_native_gather')), flush=True)
        actual = value.full_tensor().cuda()
        torch.testing.assert_close(actual, x, rtol=0, atol=0)
        print(json.dumps(dict(rank=dist.get_rank(), device=device, shape=shape, phase='passed')), flush=True)
dist.destroy_process_group()
