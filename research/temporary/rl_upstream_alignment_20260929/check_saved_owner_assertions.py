"""Apply pinned VERL assertions to saved matched actor artifacts, with scope stated."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from verl.utils.torch_functional import masked_mean

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/upstream-alignment-20260929')
source = root/'official/tests/models/test_transformer.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_hf_casual_models')
assertion = next(n for n in ast.walk(fn) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == 'torch.testing.assert_close')
check = compile(ast.fix_missing_locations(ast.Module(body=[assertion], type_ignores=[])), str(source), 'exec')
results = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    assertion=ast.unparse(assertion),
    scope='Original VERL masked-mean logprob assertion applied only before the first update, at identical weights. Later independent updates have different weights and are diagnostics, not this official same-weight test. No individual-token, gradient or whole-PPO threshold is inferred.', cases=[])
grad_source = root/'official/tests/models/test_transformers_ulysses.py'
grad_tree = ast.parse(grad_source.read_text())
grad_fn = next(n for n in grad_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_hf_casual_fwd_bwd')
grad_assertion = next(n for n in ast.walk(grad_fn) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == 'torch.testing.assert_close'
                     and ast.unparse(n.value.args[0]) == 'grad')
grad_check = compile(ast.fix_missing_locations(ast.Module(body=[grad_assertion], type_ignores=[])), str(grad_source), 'exec')
results['gradient_comparison'] = dict(source=str(grad_source),
    sha256=hashlib.sha256(grad_source.read_bytes()).hexdigest(), assertion=ast.unparse(grad_assertion),
    scope='The pinned VERL gradient assertion, unchanged, applied to first-update LoRA gradients at matched initial weights/inputs/advantages. Upstream tests a q_proj gradient under sequence parallelism; these are additional actor-seam cases, not an unchanged upstream test or a general optimizer-update tolerance.', cases=[])
for name in ('isolated-vs-owner', 'response-span-vs-owner', 'native-loss-fp32-reduction'):
    data = torch.load(root/(name+'.pt'), map_location='cpu', weights_only=True)
    actual_grad, reference_grad = data['raw_gradients'][0], data['paired_owner']['raw_gradients'][0]
    assert actual_grad.keys() == reference_grad.keys()
    failures, maximum = [], 0.0
    for key in actual_grad:
        actual, reference = actual_grad[key], reference_grad[key]
        maximum = max(maximum, float((actual-reference).abs().max()))
        try:
            exec(grad_check, dict(torch=torch, grad=actual, grad_full=reference))
        except AssertionError as exc:
            failures.append(dict(parameter=key, error=str(exc)))
    results['gradient_comparison']['cases'].append(dict(artifact=name, step=0,
        tensors=len(actual_grad), max_absolute_difference=maximum, passed=not failures, failures=failures))
    mask = data['response_mask']
    for step,(actual,reference) in enumerate(zip(data['policy_forward_log_probs'], data['paired_owner']['policy_forward_log_probs'])):
        # Owner assertion indexes a response-length slice; retain identical selected mask.
        scope = dict(torch=torch, masked_mean=masked_mean, log_probs=actual,
                     origin_log_probs=reference, attention_mask=torch.nn.functional.pad(mask,(0,1)),
                     response_length=mask.shape[-1])
        row = dict(artifact=name, step=step, shape=list(actual.shape),
                   mean_actual=float(masked_mean(actual,mask)), mean_reference=float(masked_mean(reference,mask)))
        if step > 0:
            row['scope'] = 'Different weights after independently accumulated updates; no owner assertion applied.'
            results['cases'].append(row)
            continue
        try:
            exec(check, scope)
            row['passed'] = True
        except AssertionError as exc:
            row.update(passed=False, error=str(exc))
        results['cases'].append(row)
(root/'saved-actor-official-assertions.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
