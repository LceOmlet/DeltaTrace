"""Test the original offload handler's group schedule, not a model/PPO substitute."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
p=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
if '--native' not in sys.argv:
    import re
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line) for line in smi.splitlines()), 'Physical2/3 occupied'
    env=dict(os.environ);env.update(p['effective_environment']);env['CUDA_VISIBLE_DEVICES']='2'
    env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

import copy
import hashlib
import inspect
import time
import torch
from verl.utils.activation_offload import (ActivationHandler,FSDPParameterFilter,
    get_activation_offload_context)
import verl.utils.activation_offload as owner
model_path=p['options']['actor_rollout_ref.model.path']
cfg=json.loads((Path(model_path)/'config.json').read_bytes())
active=cfg['text_config']['num_hidden_layers']
vision=cfg['vision_config']['depth']
torch.cuda.set_device(0)
torch.manual_seed(20261016)
base=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False,device='cuda',dtype=torch.bfloat16) for _ in range(active)])
with torch.no_grad():
    for layer in base:
        layer.weight.copy_(torch.eye(4,device='cuda',dtype=torch.bfloat16))
x=torch.arange(1,9,device='cuda',dtype=torch.bfloat16).reshape(2,4)

def run(declared_groups):
    layers=copy.deepcopy(base)
    filter=FSDPParameterFilter()
    ctx,sync=get_activation_offload_context(declared_groups-1,declared_groups,filter)
    handler=ActivationHandler(ctx,sync,filter,True)
    for layer in layers:
        handler.wrap_module_forward_method(layer)
    value=x.clone().requires_grad_(True)
    output=value
    for layer in layers:
        output=layer(output)
    error=None
    try:
        output.float().sum().backward()
        torch.cuda.synchronize()
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    h=ctx.offload_handler
    return dict(declared_groups=declared_groups,actually_executed_groups=active,error=error,
        final_current_group=h.current_group,offloaded_group_count=h.offloaded_group_count,
        tensor_states_still_CPU_tuples=sum(isinstance(v,tuple) for v in h.tensor_tag_to_state.values()),
        input_gradient=None if value.grad is None else value.grad.detach().cpu().tolist(),
        weight_gradients=None if error else [layer.weight.grad.detach().cpu().tolist() for layer in layers])

broken=run(active+vision)
fixed=run(active)
record=dict(collected_unix=time.time(),scope=__doc__,source=dict(path=owner.__file__,
    SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()),
    inactive_vision_groups=vision,active_text_groups=active,full_model_groups=active+vision,
    original_handler_full_model_count=broken,original_handler_active_text_count=fixed,
    LLM_calls=0,PPO_calls=0,optimizer_updates=0,formal_training_mutations=0,
    limitation='Tiny original-handler state-machine reproduction only; no claim of full critic numerical equivalence, 32k capacity or efficiency.')
(AUDIT/'native-offload-group-reproduction.json').write_text(json.dumps(record,indent=2)+'\n')
assert broken['error'] is not None and broken['error']['type']=='AssertionError'
assert fixed['error'] is None
print(json.dumps({k:v for k,v in record.items() if k not in ('original_handler_full_model_count','original_handler_active_text_count')},indent=2))
print(json.dumps(dict(full_model_count_result={k:v for k,v in broken.items() if k!='weight_gradients'},
    active_text_count_result={k:v for k,v in fixed.items() if k!='weight_gradients'}),indent=2))
