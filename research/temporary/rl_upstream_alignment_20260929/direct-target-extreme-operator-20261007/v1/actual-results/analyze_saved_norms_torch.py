"""Describe real saved norm operands using original DT and Torch RMSNorm on CPU."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=root/'receipts/direct-target-extreme-operator-20261007-v1/results'
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes());env=source['environment'];dt=Path(env['DT_ROOT'])
sys.path[:0]=[str(dt/'clean/qwen35'),*source['pythonpath'].split(':')]
from signed_secant_rules import rmsnorm_secant_pullback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(4);records=[]
for name in ('post_norm','input_norm'):
 p=out/('rank0-decoder27-'+name+'.pt');d=torch.load(p,map_location='cpu',weights_only=False)
 with torch.no_grad():
  finite=rmsnorm_secant_pullback(d['x0'].float(),d['x1'].float(),1+d['weight'].float(),d['upstream'],d['eps'])+d['residual']
  delta=(d['x1'].double()-d['x0'].double())
  input_effect=float((finite.double()*delta).sum());saved_input_effect=float((d['returned'].double()*delta).sum())
  residual_effect=float((d['residual'].double()*delta).sum())
  single_delta=d['single_input'][1:2].double()-d['single_input'][0:1].double()
  single_effect=float((finite.double()*single_delta).sum());saved_single=float((d['returned'].double()*single_delta).sum())
  # Same original finite norm evaluated in FP64: description, no new rule.
  f64=rmsnorm_secant_pullback(d['x0'].double(),d['x1'].double(),1+d['weight'].double(),d['upstream'].double(),d['eps'])+d['residual'].double()
  continuous0=torch.nn.functional.rms_norm(d['x0'].double(),[d['x0'].shape[-1]],1+d['weight'].double(),d['eps'])
  continuous1=torch.nn.functional.rms_norm(d['x1'].double(),[d['x1'].shape[-1]],1+d['weight'].double(),d['eps'])
  continuous_effect=float((d['upstream'].double()*(continuous1-continuous0)).sum())+residual_effect
  finite64_effect=float((f64*delta).sum())
 row=dict(name=name,path=str(p),sha256=sha(p),dtypes={k:str(v.dtype) for k,v in d.items() if isinstance(v,torch.Tensor)},compiled_vs_original_eager_coefficient_maxabs=float((finite-d['returned']).abs().max()),joint_compiled_input_effect=saved_input_effect,joint_original_eager_input_effect=input_effect,joint_continuous_fp64_effect=continuous_effect,joint_finite_fp64_effect=finite64_effect,fp64_endpoint_identity_residual=finite64_effect-continuous_effect,single_direction_compiled_effect=saved_single,single_direction_original_eager_effect=single_effect,single_direction_original_fp64_effect=float((f64*single_delta).sum()))
 records.append(row)
assert not torch.cuda.is_initialized()
r=dict(scope=__doc__,owners=dict(Torch_reference=dict(function='torch.nn.functional.rms_norm',path=inspect.getsourcefile(torch.nn.functional.rms_norm),sha256=sha(inspect.getsourcefile(torch.nn.functional.rms_norm))),finite=dict(path=inspect.getsourcefile(rmsnorm_secant_pullback),sha256=sha(inspect.getsourcefile(rmsnorm_secant_pullback)))),records=records,acceptance='Descriptive actual-operand computation only. No new tolerance, correction or production operation.')
(out/'norm-torch-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
