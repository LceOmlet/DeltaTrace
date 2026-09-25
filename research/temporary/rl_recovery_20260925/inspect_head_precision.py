"""Same saved head operands; original linear/log-prob owners at higher precision.

CPU arithmetic diagnosis only. No model forward, DT rule, reward or production
setting is changed; a small head residual is not proof of token accuracy.
"""
import json
import importlib.util
import os
from pathlib import Path
import torch
from compiled_logprob_seed import seed_with_checks

torch.set_num_threads(4)
r = Path(os.environ['DT_RUNTIME_ROOT'])
baseline = r/'releases/c88a749/clean/qwen35/qwen35_answer_finite.py'
spec = importlib.util.spec_from_file_location('head_rounding_baseline', baseline)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
source = r/'receipts/dt-extremes-side-20260925-1027/fixed-input-trace-0.pt'
saved = torch.load(source, map_location='cpu', weights_only=True)
z0, z1, target, weight, h0, h1 = saved['head']
old = owner._rounded_answer_seed_rule(z0, z1, target, weight, h0, h1)[0].double()
reference = [torch.nn.functional.linear(h.double(), weight.double()) for h in (h0, h1)]
projected = [torch.nn.functional.linear(h.float(), weight.float()) for h in (h0, h1)]
seed, checks = seed_with_checks(*projected, target)
assert bool(checks)
coefficient = seed @ weight.float()
delta = h1.double()-h0.double()
root64 = (reference[1].log_softmax(-1)-reference[0].log_softmax(-1)).gather(-1,target[:,None]).squeeze(-1)
root32 = (projected[1].log_softmax(-1)-projected[0].log_softmax(-1)).gather(-1,target[:,None]).squeeze(-1)
native = (z1.float().log_softmax(-1)-z0.float().log_softmax(-1)).gather(-1,target[:,None]).squeeze(-1)
rows=[]
for i,event in enumerate((10,11,12,13)):
    effect=(coefficient[i].double()*delta[i]).sum()
    rows.append(dict(event=event,fp64_root=float(root64[i]),fp32_root=float(root32[i]),
        native_bf16_root=float(native[i]),fp32_root_abs_error=float(abs(root32[i]-root64[i])),
        bf16_root_abs_error=float(abs(native[i]-root64[i])),
        original_head_coefficient_norm=float(old[i].norm()),
        fp32_linear_head_coefficient_norm=float(coefficient[i].norm()),
        fp32_finite_effect=float(effect),fp32_finite_residual=float(effect-root32[i]),
        fp32_absolute_effect_sum=float((coefficient[i].double()*delta[i]).abs().sum())))
result=dict(scope=__doc__,source=str(source),baseline_owner=str(baseline),rows=rows,full_model_or_token_repair_tested=False)
(r/'receipts/rollout-major-cost/native-head-fp32-operand-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
