"""Native handler regression for exposed Torch checkpoint mode; not full PPO validation."""
import os
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
BASE=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
CANDIDATE=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v1/verl-owner'
p=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
if '--native' not in sys.argv:
    import re
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line) for line in smi.splitlines()),'Physical2/3 occupied'
    env=dict(os.environ);env.update(p['effective_environment'])
    env['PYTHONPATH']=str(CANDIDATE)+':'+env['PYTHONPATH']
    env['VERL_ROOT']=str(CANDIDATE);env['CUDA_VISIBLE_DEVICES']='2'
    env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

import copy
import hashlib
import torch
import verl.utils.activation_offload as owner
from verl.utils.activation_offload import ActivationHandler,FSDPParameterFilter,get_activation_offload_context
torch.cuda.set_device(0);torch.manual_seed(20261016)
layers=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False,device='cuda',dtype=torch.bfloat16) for _ in range(32)])
with torch.no_grad():
    for layer in layers:layer.weight.copy_(torch.eye(4,device='cuda',dtype=torch.bfloat16))
x=torch.arange(1,9,device='cuda',dtype=torch.bfloat16).reshape(2,4)

def run(mode):
    model=copy.deepcopy(layers)
    if mode!='plain':
        filt=FSDPParameterFilter();ctx,sync=get_activation_offload_context(31,32,filt)
        kwargs={} if mode=='default' else dict(use_reentrant=False)
        handler=ActivationHandler(ctx,sync,filt,True,**kwargs)
        for layer in model:handler.wrap_module_forward_method(layer)
    value=x.clone().requires_grad_(True);y=value
    for layer in model:y=layer(y)
    y.float().square().mean().backward();torch.cuda.synchronize()
    return dict(output=y.detach().cpu(),input_gradient=value.grad.detach().cpu(),
                weight_gradients=torch.stack([layer.weight.grad.detach().cpu() for layer in model]))

reference=run('plain');results={}
for mode in ('default','non_reentrant'):
    actual=run(mode)
    for key,value in actual.items():torch.testing.assert_close(value,reference[key],rtol=0.0,atol=0.0)
    results[mode]='exactly_equal_to_plain'
record=dict(scope=__doc__,results=results,numerical_criterion='Native memory-copy regression: rtol=0, atol=0',
            source=dict(path=owner.__file__,SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()),
            original_source_bytes_SHA256=hashlib.sha256((BASE/'verl/utils/activation_offload.py').read_bytes()).hexdigest(),
            LLM_calls=0,PPO_calls=0,optimizer_updates=0,formal_mutations=0)
(AUDIT/'candidate-handler-regression.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
