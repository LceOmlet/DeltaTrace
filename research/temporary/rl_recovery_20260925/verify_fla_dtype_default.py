"""Check the unchanged BF16 path on actual unequal endpoint captures.

Both calls use the existing finite algorithm and original FLA stages. The
candidate merely selects GEMM dtype from the captured native q tensor.
"""
import importlib.util
import json
import os
from pathlib import Path
import torch

root=Path(os.environ['DT_RUNTIME_ROOT'])
rec=root/'receipts/rollout-major-cost'
saved=torch.load(rec/'native-prefix-gdn-operands.pt',map_location='cpu',weights_only=True,mmap=True)
part=saved['cached_gdn']
endpoints={n:x.cuda() for n,x in part['endpoints'].items() if n!='o'}
torch.manual_seed(42)
upstream=torch.randn_like(endpoints['v'][1::2])
results=[]
for name,path in [('baseline',root/'releases/c88a749/clean/qwen35/finite_fla_gpu.py'),
                  ('candidate',root/'candidates/native-fla-fp16-20260925/clean/qwen35/finite_fla_gpu.py')]:
    spec=importlib.util.spec_from_file_location(name+'_finite_fla',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        results.append(module.finite_fla_pullback(endpoints,upstream,part['scale']))
checks={n:dict(equal=torch.equal(results[0][n],results[1][n]),
               max_abs=float((results[0][n]-results[1][n]).abs().max())) for n in results[0]}
out=dict(scope=__doc__,shape=list(endpoints['q'].shape),checks=checks)
(rec/'native-fla-dtype-default.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
assert all(x['equal'] for x in checks.values())
