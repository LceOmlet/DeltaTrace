"""Native Torch collective preflight for the requested two-card VERL workers."""
import json
import os
import torch

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')
value = torch.tensor([torch.distributed.get_rank() + 1.0], device='cuda')
torch.distributed.all_reduce(value)
assert value.item() == 3.0
print(json.dumps(dict(rank=torch.distributed.get_rank(), device=rank,
                      device_name=torch.cuda.get_device_name(rank), sum=value.item())), flush=True)
torch.distributed.destroy_process_group()
