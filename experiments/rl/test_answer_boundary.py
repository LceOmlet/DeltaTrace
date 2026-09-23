"""Numerical regressions for the native categorical head's rounding boundary.

Import the formal DT owner via the recorded runtime PYTHONPATH. These use
actual torch Linear outputs, never an invented attribution vector.
"""
import pytest
import torch

from qwen35_answer_finite import FiniteAnswerOps, PackedAnswerTargets


def operands(step=3.0, target=0):
    head = torch.nn.Linear(3, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[.2, .04, 0.], [.2, -.04, 0.], [.2, 0., .04]]))
    hidden = torch.tensor([[100., 0., 0.], [100., step, 0.]], dtype=torch.bfloat16)
    logits = head(hidden).detach()
    targets = PackedAnswerTargets(
        [{'target_ids': torch.tensor([target]), 'prompt_length': 1}], [[0]], 2, 'cpu',
        outcome_token_ids=[0, 1, 2],
    )
    return head, hidden, logits, targets


@pytest.mark.parametrize('step', [1., 3., -3.])
@pytest.mark.parametrize('target', [0, 1, 2])
def test_native_bf16_head_rounding_finite_identity(step, target):
    head, hidden, logits, targets = operands(step, target)
    original_logits, original_hidden, original_weight = logits.clone(), hidden.clone(), head.weight.clone()
    with torch.no_grad():
        _, detail = FiniteAnswerOps(compiled=False)(
            logits, head, targets, original_packed_hidden=hidden,
        )
    contribution = (detail['packed_hidden'].double() * (hidden[1:].double()-hidden[:1].double())).sum()
    native = logits.float().log_softmax(-1)
    effect = (native[1, target]-native[0, target]).double()
    torch.testing.assert_close(contribution, effect, atol=2e-5, rtol=1e-5)
    assert torch.equal(logits, original_logits)
    assert torch.equal(hidden, original_hidden)
    assert torch.equal(head.weight, original_weight)
    if step == 1:
        # W delta(h) is nonzero, but the native BF16 head rounds both
        # endpoints to the same logits. The former linear rule missed this.
        assert (torch.nn.functional.linear(hidden[1:].float()-hidden[:1].float(), head.weight.float()) != 0).any()
        assert torch.equal(logits[0], logits[1])
        assert effect == 0


def test_equal_head_endpoints_have_zero_finite_effect():
    head, hidden, logits, targets = operands()
    with torch.no_grad():
        dense, detail = FiniteAnswerOps(compiled=False)(
            logits, head, targets, equal_endpoint=True, original_packed_hidden=hidden,
        )
    assert not dense.any()
    assert torch.equal(detail['logp0'], detail['logp1'])


def test_categorical_head_requires_captured_native_input():
    head, _, logits, targets = operands()
    with pytest.raises(ValueError, match='captured native input'):
        FiniteAnswerOps(compiled=False)(logits, head, targets)


def test_zero_linear_displacement_cannot_hide_changed_native_output():
    head, hidden, logits, targets = operands()
    hidden[1] = hidden[0]
    with pytest.raises(ValueError, match='Finite answer seed invalid'):
        FiniteAnswerOps(compiled=False)(logits, head, targets, original_packed_hidden=hidden)


def test_cached_suffix_preserves_target_identity_and_endpoint_packing():
    targets=PackedAnswerTargets([
        {'prompt_length':9,'target_ids':torch.tensor([3,4])},
        {'prompt_length':11,'target_ids':torch.tensor([5])},
    ],[[0,1],[0]],12,'cpu')
    hidden=torch.arange(4*12*3).reshape(4,12,3)
    suffix=targets.suffix(8)
    assert torch.equal(targets.pack_hidden(hidden),suffix.pack_hidden(hidden[:,8:]))
    assert suffix.labels is targets.labels and suffix.paired_samples is targets.paired_samples
    assert targets.length==12 and targets.positions.tolist()==[8,9,10]
    packed=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    assert torch.equal(targets.scatter_hidden(packed)[:,8:],suffix.scatter_hidden(packed))
    with pytest.raises(ValueError,match='predictor'):
        targets.suffix(9)
