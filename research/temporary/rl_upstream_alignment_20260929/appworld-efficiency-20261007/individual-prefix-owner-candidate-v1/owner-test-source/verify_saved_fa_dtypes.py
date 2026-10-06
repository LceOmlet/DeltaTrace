"""Apply pinned FA references and unmodified assertions to recorded DT operands.

The finite comparison uses coincident factual endpoints and only the coefficient
suffix computed by the deployed owner. It is not a nonzero-finite-error bound.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import torch
from flash_attn import flash_attn_func
from vendor_fa_finite_bf16_d256 import RightPaddedLengths, VendorFAFiniteP1BF16D256
from verify_official_kernel_tolerances import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--operands', type=Path, required=True)
parser.add_argument('--sources', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
source = args.sources/'test_flash_attn_v263.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
test = load('saved_operand_fa_reference', source)
function = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'test_flash_attn_output')
assertions = {}
for node in ast.walk(function):
    if isinstance(node, ast.Assert):
        expression = ast.unparse(node.test)
        for name in ('out', 'dq', 'dk', 'dv'):
            if expression.startswith(f'({name} - {name}_ref).abs().max().item() <='):
                assertions[name] = (compile(ast.Module(body=[node], type_ignores=[]),str(source),'exec'), expression)
assert set(assertions) == {'out','dq','dk','dv'}
saved = torch.load(args.operands,map_location='cpu',weights_only=True,mmap=True)
fa = saved['fa']
a = saved['attention_values']
q,k,v = [a[name][1::2].cuda().detach().requires_grad_() for name in ('dense_q','dense_k','dense_v')]
g = fa['operands']['u'].transpose(1,2).cuda()
print('phase=native_and_reference shape='+str(list(q.shape)),flush=True)
out,lse,_ = flash_attn_func(q,k,v,causal=True,deterministic=True,return_attn_probs=True)
ref,_ = test.attention_ref(q,k,v,causal=True)
pt,_ = test.attention_ref(q,k,v,causal=True,upcast=False,reorder_ops=True)
native_grads = torch.autograd.grad(out,(q,k,v),g)
ref_grads = torch.autograd.grad(ref,(q,k,v),g)
pt_grads = torch.autograd.grad(pt,(q,k,v),g)
env = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_text())['qwen35']
owner = VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
layout = RightPaddedLengths(list(fa['lengths']),fa['padded_length'],q.device,
    coefficient_starts=list(fa['coefficient_starts']) if fa['coefficient_starts'] is not None else None,
    query_start=fa['query_start'])
ops = dict(q0=q.transpose(1,2),q1=q.transpose(1,2),k0=k.transpose(1,2),k1=k.transpose(1,2),
           v0=v.transpose(1,2),u=g.transpose(1,2),lse0=lse,lse1=lse)
with torch.no_grad():
    finite = owner(ops,fa['scale'],layout)
batch,heads,time,dim = finite['dq'].shape
kv_heads = k.shape[2]
def reduced(name):
    return finite[name].float().reshape(batch,kv_heads,heads//kv_heads,time,dim).sum(2).transpose(1,2).to(k.dtype)
finite_grads = (finite['dq'].transpose(1,2),reduced('dk'),reduced('dv'))
starts = torch.tensor(fa['coefficient_starts'] or [fa['query_start']]*batch,device=q.device)
positions = torch.arange(time,device=q.device)+fa['query_start']
mask = positions[None,:]>=starts[:,None]
result = dict(scope=__doc__,source=str(args.operands),query_shape=list(q.shape),key_shape=list(k.shape),
    operand_dtypes=dict(q=str(q.dtype),k=str(k.dtype),v=str(v.dtype),upstream=str(g.dtype),lse=str(lse.dtype)),
    native_kernel_upstream_dtype=str(out.dtype), finite_kernel_upstream_dtype=str(q.dtype),
    reference='pinned attention_ref, FP32 upcast; original reordered low precision baseline',
    assertions={name:item[1] for name,item in assertions.items()},query_start=fa['query_start'],
    coefficient_starts=fa['coefficient_starts'],checks=[])
def check(name,actual,expected,baseline,kind):
    context={name:actual,name+'_ref':expected,name+'_pt':baseline}
    row=dict(name=name,owner=kind,actual_dtype=str(actual.dtype),reference_dtype=str(expected.dtype),
             max_abs=float((actual-expected).abs().max()),baseline_max_abs=float((baseline-expected).abs().max()))
    try:
        exec(assertions[name][0],{},context)
        row['status']='passed'
    except AssertionError:
        row['status']='failed'
    result['checks'].append(row)
check('out',out,ref,pt,'native')
for i,name in enumerate(('dq','dk','dv')):
    reference = ref_grads[i] if i==0 else ref_grads[i][:,fa['query_start']:]
    baseline = pt_grads[i] if i==0 else pt_grads[i][:,fa['query_start']:]
    native = native_grads[i] if i==0 else native_grads[i][:,fa['query_start']:]
    check(name,native[mask],reference[mask],baseline[mask],'native')
    check(name,finite_grads[i][mask],reference[mask],baseline[mask],'finite')
result['status']='passed' if all(row['status']=='passed' for row in result['checks']) else 'failed'
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
