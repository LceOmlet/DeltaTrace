"""Bounded reproduction of the observed MetaX host-register failure, no model."""
import gc
import json
import os
from pathlib import Path
import sys
import time

import psutil
import torch

mode=sys.argv[1]
assert mode in ('True','False')
block_count=int(sys.argv[3]) if len(sys.argv)>3 else 16
assert 1<=block_count<=16
assert ('pinned_use_cuda_host_register:'+mode) in os.environ['PYTORCH_CUDA_ALLOC_CONF']
assert psutil.virtual_memory().available > 140*2**30
torch.cuda.init()
blocks=[]
attempts=[]
error=None
tick=time.monotonic()
try:
    source=torch.arange(4096,device='cuda',dtype=torch.int32)
    for index in range(block_count):
        started=time.monotonic()
        block=torch.empty(6*2**30,dtype=torch.uint8,pin_memory=True)
        blocks.append(block)
        check=block[:4096*4].view(torch.int32)
        check.copy_(source,non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(check,torch.arange(4096,dtype=torch.int32))
        attempts.append(dict(live_GiB=6*len(blocks),allocation_and_copy_seconds=time.monotonic()-started))
except Exception as exc:
    error=dict(type=type(exc).__name__,message=str(exc))
finally:
    del block,check
    blocks.clear()
    gc.collect()
    torch._C._host_emptyCache()
record=dict(mode=mode,status='native_allocation_failed' if error else f'passed_{6*block_count}GiB_native_pinned_allocation_copy',
    attempts=attempts,error=error,seconds=time.monotonic()-tick,pid=os.getpid(),
    birth=psutil.Process().create_time(),model_calls=0,DT_calls=0,optimizer_updates=0,
    scope='Only the observed pinned allocation interface; not training capacity or throughput')
path=Path(sys.argv[2]);path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
