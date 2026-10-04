"""CPU FP32 interface algebra against actual PEFT, not GPU/FA acceptance.

Execute the actual candidate functions, using native CPU FP32 torch.bmm only
in place of the owner's GPU-only BF16/out_dtype MM. Upstream values are exactly
BF16-representable, so the existing operand cast does not alter this algebra
fixture. No production precision setting, formula or PEFT forward is changed.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from peft.tuners.lora.layer import Linear


AUDIT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_isolated_lora_preparer",
    AUDIT/"prepare_native_lora_factorization_owner_20261004.py")
PREPARER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARER)
OWNER_PATH = Path(os.environ.get("DT_LORA_FACTORIZATION_OWNER_SOURCE",PREPARER.OWNER))


@pytest.fixture(scope="module")
def candidate():
    original=OWNER_PATH.read_bytes()
    source,_=PREPARER.patched_owner(original)
    functions=[n for n in ast.parse(source).body
               if isinstance(n,ast.FunctionDef) and n.name in PREPARER.REPLACEMENTS]
    calls=[]
    def cpu_fp32_mm(left,right):
        assert left.device.type==right.device.type=='cpu'
        calls.append((tuple(left.shape),tuple(right.shape)))
        return torch.bmm(left.float(),right.float())
    namespace=dict(torch=torch,_mm=cpu_fp32_mm)
    exec(compile(ast.Module(body=functions,type_ignores=[]),
         "isolated_qwen35_decoder_finite_low_rank_candidate.py","exec"),namespace)
    return namespace,calls,source


def actual_layer():
    torch.manual_seed(104)
    base=torch.nn.Linear(16,12,bias=False,dtype=torch.float32,device='cpu')
    layer=Linear(base,'default',r=8,lora_alpha=16,lora_dropout=0.,fan_in_fan_out=False)
    with torch.no_grad():
        layer.lora_A['default'].weight.normal_(std=.15)
        layer.lora_B['default'].weight.normal_(std=.2)
    return layer.eval()


@pytest.mark.parametrize('state',['active','disabled','merged'])
def test_actual_peft_fp32_input_pullback_and_unchanged_parameters(candidate,state):
    functions,calls,_=candidate
    layer=actual_layer()
    if state=='disabled':
        layer.enable_adapters(False)
    elif state=='merged':
        layer.merge()
    before={name:value.clone() for name,value in layer.state_dict().items()}
    torch.manual_seed(105)
    inputs=torch.randn(4,3,16,dtype=torch.float32,requires_grad=True)
    # Nonuniform signed per-token adjoints, all exactly BF16 representable.
    upstream=torch.randint(-7,8,(4,3,12)).float()/8
    expected=torch.autograd.grad(layer(inputs),inputs,grad_outputs=upstream)[0]
    calls.clear()
    with torch.no_grad():
        if state=='active':
            # The replacement must not materialize a full delta matrix.
            with patch.object(layer,'get_delta_weight',side_effect=AssertionError('unexpected full delta')):
                weights=functions['_linear_weights'](layer)
        else:
            weights=functions['_linear_weights'](layer)
        actual=functions['_linear_transpose'](upstream,weights)
    torch.testing.assert_close(actual,expected)  # Original torch FP32 default, not a new GPU tolerance.
    if state=='active':
        assert isinstance(weights,tuple) and len(weights)==2
        adapter_B,adapter_A,scale=weights[1]
        assert weights[0] is layer.weight
        assert adapter_B is layer.lora_B['default'].weight
        assert adapter_A is layer.lora_A['default'].weight
        assert scale==layer.scaling['default']==2
        assert calls==[((1,12,12),(1,12,16)),((1,12,12),(1,12,8)),((1,12,8),(1,8,16))]
        assert not torch.equal(expected,upstream@layer.weight)
    else:
        assert weights is layer.weight
        assert calls==[((1,12,12),(1,12,16))]
    for name,value in layer.state_dict().items():
        assert torch.equal(value,before[name]),name


def test_plain_linear_preserves_original_weight_and_map(candidate):
    functions,calls,_=candidate
    layer=torch.nn.Linear(16,12,bias=True,dtype=torch.float32)
    upstream=torch.randint(-3,4,(4,2,12)).float()/4
    inputs=torch.randn(4,2,16,requires_grad=True)
    expected=torch.autograd.grad(layer(inputs),inputs,grad_outputs=upstream)[0]
    assert functions['_linear_weights'](layer) is layer.weight
    calls.clear()
    actual=functions['_linear_transpose'](upstream,layer.weight)
    torch.testing.assert_close(actual,expected)
    assert calls==[((1,8,12),(1,12,16))]


def test_actual_peft_active_adapter_selection_uses_owner_factors(candidate):
    functions,calls,_=candidate
    layer=actual_layer()
    layer.update_layer('second',r=8,lora_alpha=16,lora_dropout=0.,init_lora_weights=True,
                       use_rslora=False,use_dora=False,lora_bias=False)
    with torch.no_grad():
        layer.lora_A['second'].weight.normal_(std=.1)
        layer.lora_B['second'].weight.normal_(std=.1)
    layer.set_adapter(['default','second'])
    upstream=torch.randint(-3,4,(4,2,12)).float()/4
    inputs=torch.randn(4,2,16,requires_grad=True)
    expected=torch.autograd.grad(layer(inputs),inputs,grad_outputs=upstream)[0]
    with patch.object(layer,'get_delta_weight',side_effect=AssertionError('unexpected full delta')):
        weights=functions['_linear_weights'](layer)
    assert len(weights)==3 and weights[2][0] is layer.lora_B['second'].weight
    calls.clear()
    actual=functions['_linear_transpose'](upstream,weights)
    torch.testing.assert_close(actual,expected)
    assert len(calls)==5
    layer.set_adapter('second')
    weights=functions['_linear_weights'](layer)
    assert len(weights)==2 and weights[1][0] is layer.lora_B['second'].weight


def test_outside_two_owner_ranges_and_original_base_paths_preserved(candidate):
    _,_,source=candidate
    original=OWNER_PATH.read_bytes()
    assert hashlib.sha256(original).hexdigest()==PREPARER.OWNER_SHA
    original_tree=ast.parse(original);candidate_tree=ast.parse(source)
    extract=lambda tree:[ast.dump(n) for n in tree.body
        if not (isinstance(n,ast.FunctionDef) and n.name in PREPARER.REPLACEMENTS)]
    assert extract(original_tree)==extract(candidate_tree)
    old=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_linear_transpose')
    new=next(n for n in candidate_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_linear_transpose')
    assert [ast.dump(n) for n in old.body[1:]]==[ast.dump(n) for n in new.body[1:]]
    # The base operand preparation and first owner MM call remain identical.
    assert [ast.dump(n) for n in old.body[0].body[:3]]==[ast.dump(n) for n in new.body[0].body[:3]]
