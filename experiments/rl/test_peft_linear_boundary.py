"""Formal DT weight readout against the installed PEFT linear owner."""
import pytest
import torch
from peft.tuners.lora.layer import Linear
from qwen35_decoder_finite import _linear_weights, _linear_transpose


def adapter(dtype=torch.float64, device='cpu'):
    base=torch.nn.Linear(16,8,bias=False,dtype=dtype,device=device)
    layer=Linear(base,'default',r=2,lora_alpha=4,lora_dropout=0.)
    with torch.no_grad():
        base.weight.fill_(.125)
        layer.lora_A['default'].weight.fill_(.0625)
        layer.lora_B['default'].weight.fill_(.125)
    return layer.eval()


@pytest.mark.parametrize('state',['active','disabled','merged'])
def test_peft_weight_readout_matches_native_input_gradient_without_mutation(state):
    layer=adapter()
    if state=='disabled':
        layer.enable_adapters(False)
    elif state=='merged':
        layer.merge()
    before={name:value.clone() for name,value in layer.state_dict().items()}
    x=torch.arange(16,dtype=torch.float64).reshape(1,16).requires_grad_()
    expected=torch.autograd.grad(layer(x).sum(),x)[0]
    with torch.no_grad():
        weights=_linear_weights(layer)
        parts=weights if isinstance(weights,tuple) else (weights,)
        actual=sum(torch.ones(1,8,dtype=torch.float64) @ w for w in parts)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    if state=='active':
        assert isinstance(weights,tuple)
        assert not torch.equal(expected,torch.ones(1,8,dtype=torch.float64) @ layer.weight)
    else:
        assert weights is layer.weight
    for name,value in layer.state_dict().items():
        assert torch.equal(value,before[name])


def test_plain_linear_keeps_original_weight_object():
    layer=torch.nn.Linear(16,8)
    assert _linear_weights(layer) is layer.weight


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native BF16 GEMM requires GPU')
def test_native_and_compiled_finite_gemm_preserve_small_unmerged_adapter():
    layer=adapter(torch.bfloat16,'cuda')
    with torch.no_grad():
        layer.get_base_layer().weight.fill_(1.)
        layer.lora_B['default'].weight.fill_(.0078125)
        weights=_linear_weights(layer)
        # BF16 pre-merging would erase this actual nonzero adapter weight.
        assert torch.equal(weights[0]+weights[1],weights[0])
        upstream=torch.ones(1,3,8,device='cuda')
        expected=sum(upstream @ w.float() for w in weights)
        actual=_linear_transpose(upstream,weights)
        compiled=torch.compile(_linear_transpose,dynamic=True)
        compiled_actual=compiled(upstream,weights)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(compiled_actual,expected,atol=1e-6,rtol=1e-6)
        assert not torch.equal(actual,_linear_transpose(upstream,weights[0]))
