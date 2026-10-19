"""Exercise the actual capture-copy owner; no model, DT, backward or optimizer."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

owner=Path(os.environ['DT_ROOT'])/'clean/qwen35/native_attention_capture.py'
spec=importlib.util.spec_from_file_location('original_native_capture_copy',owner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=torch.empty((4,32768,12288),device='cuda',dtype=torch.bfloat16)
source.copy_(torch.arange(12288,device='cuda',dtype=torch.float32).remainder(97))
torch.cuda.synchronize()
rows=[]
for label,value in [('contiguous',source),('noncontiguous',source.transpose(1,2))]:
    for attempt in range(2):
        started=time.perf_counter()
        copied=module.copy_capture_tensor(value,'cpu',pinned_host=True,preserve_strides=True)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        assert copied.is_pinned() and copied.stride()==value.stride()
        assert torch.equal(copied.to('cuda'),value)
        rows.append(dict(layout=label,attempt=attempt,bytes=value.numel()*value.element_size(),
            allocation_and_D2H_seconds=seconds,exact_values=True,original_strides=True))
        del copied
        gc.collect()
torch._C._host_emptyCache()
record=dict(status='passed_original_owner_B4_32768_BF16_copy',configuration=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
    physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],rows=rows,model_calls=0,DT_calls=0,optimizer_updates=0)
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
