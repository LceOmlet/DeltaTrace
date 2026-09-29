"""CPU FP64 accounting for the first saved native projection differences.

The recorded actor is freshly initialized (LoRA B=0). Original checkpoint
projection weights suffice; no correction or production operator is installed.
"""
import json
import os
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
model = Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B')
index = json.loads((model / 'model.safetensors.index.json').read_text())
result = dict(scope=__doc__, cases=[])
started = time.perf_counter()
for side in ('factual','reference'):
    path = audit / f'duplicate-first-difference-2-{side}.pt'
    saved = torch.load(path,map_location='cpu',weights_only=True)
    layer, module = saved['name'].split('.',1)
    key = f'model.language_model.layers.{layer}.{module}.weight'
    with safe_open(model / index['weight_map'][key],framework='pt',device='cpu') as shard:
        weight = shard.get_tensor(key)
    x, actual = saved['input'], saved['output']
    expected = F.linear(x.double(),weight.double())
    selected = slice(3,None,2) if side=='factual' else slice(2,None,2)
    xx, yy, rr = x[selected], actual[selected], expected[selected]
    different = (yy != yy[:1]).nonzero()
    case = dict(side=side,module=saved['name'],shape=list(x.shape),
        input_dtype=str(x.dtype),weight_dtype=str(weight.dtype),output_dtype=str(actual.dtype),
        input_duplicate_max_abs=float((xx.float()-xx[:1].float()).abs().max()),
        output_duplicate_max_abs=float((yy.float()-yy[:1].float()).abs().max()),
        fp64_duplicate_max_abs=float((rr-rr[:1]).abs().max()),
        max_abs_native_vs_fp64=float((yy.double()-rr).abs().max()),
        max_abs_native_vs_correctly_rounded_bf16=float((yy.float()-rr.bfloat16().float()).abs().max()),
        mismatched_duplicate_coordinates=int(different.shape[0]),examples=[])
    for batch, token, channel in different[:12].tolist():
        case['examples'].append(dict(row=batch,token=token,channel=channel,
            actual=yy[:,token,channel].float().tolist(),
            cpu_fp64=rr[:,token,channel].tolist(),
            cpu_fp64_rounded_bf16=rr[:,token,channel].bfloat16().float().tolist()))
    result['cases'].append(case)
result['seconds']=time.perf_counter()-started
(audit/'native-first-projection-fp64.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
