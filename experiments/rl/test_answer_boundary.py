"""Numerical regressions for the categorical head's FP32 readout boundary.

Import the formal DT owner via the recorded runtime PYTHONPATH. These use
actual torch Linear outputs, never an invented attribution vector.
"""
import pytest
import torch

from qwen35_answer_finite import (FiniteAnswerOps, PackedAnswerTargets,
    categorical_head_logits, selected_target_log_probs, _answer_seed_rule)


def operands(step=3.0, target=0, device='cpu', classes=3):
    head = torch.nn.Linear(3, classes, bias=False, dtype=torch.bfloat16,device=device)
    with torch.no_grad():
        weights = torch.tensor([[.2, .04, 0.], [.2, -.04, 0.], [.2, 0., .04]])
        head.weight.copy_(weights.repeat((classes + 2) // 3, 1)[:classes])
    hidden = torch.tensor([[100., 0., 0.], [100., step, 0.]], dtype=torch.bfloat16,device=device)
    logits = head(hidden).detach()
    targets = PackedAnswerTargets(
        [{'target_ids': torch.tensor([target]), 'prompt_length': 1}], [[0]], 2, device,
        outcome_token_ids=list(range(classes)),
    )
    return head, hidden, logits, targets


@pytest.mark.parametrize('step', [1., 3., -3.])
@pytest.mark.parametrize('classes,target', [(3,0),(3,1),(3,2),(2,0),(2,1),(31,0),(31,16),(31,30)])
def test_categorical_readout_and_seed_use_same_fp32_head(step, classes, target):
    head, hidden, logits, targets = operands(step, target, classes=classes)
    original_logits, original_hidden, original_weight = logits.clone(), hidden.clone(), head.weight.clone()
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
        event_logits=categorical_head_logits(head,hidden,targets.outcome_token_ids)
        _, detail = FiniteAnswerOps(compiled=False)(
            logits, head, targets, original_packed_hidden=hidden,outcome_logits=event_logits,
        )
    assert event_logits.dtype == torch.float32
    assert detail['packed_hidden'].dtype == torch.float32
    contribution = (detail['packed_hidden'].double() * (hidden[1:].double()-hidden[:1].double())).sum()
    reference = torch.nn.functional.linear(hidden.double(),head.weight.double()).log_softmax(-1)
    effect = reference[1,target]-reference[0,target]
    torch.testing.assert_close(contribution, effect, atol=2e-5, rtol=1e-5)
    root=selected_target_log_probs(logits,targets,outcome_logits=event_logits)
    assert torch.equal(root[0::2],detail['logp0'])
    assert torch.equal(root[1::2],detail['logp1'])
    torch.testing.assert_close((root[1]-root[0]).double(),effect,atol=2e-5,rtol=1e-5)
    assert torch.equal(logits, original_logits)
    assert torch.equal(hidden, original_hidden)
    assert torch.equal(head.weight, original_weight)
    if step == 1:
        # Native BF16 erases a real input effect. The event readout retains it,
        # without changing or pretending to reproduce the stored BF16 logits.
        assert (torch.nn.functional.linear(hidden[1:].float()-hidden[:1].float(), head.weight.float()) != 0).any()
        assert torch.equal(logits[0], logits[1])
        assert effect != 0


def test_equal_head_endpoints_have_zero_finite_effect():
    head, hidden, logits, targets = operands()
    with torch.no_grad():
        dense, detail = FiniteAnswerOps(compiled=False)(
            logits, head, targets, equal_endpoint=True, original_packed_hidden=hidden,
        )
    assert torch.isfinite(dense).all()
    assert not (detail['packed_hidden']*(hidden[1:]-hidden[1:])).any()
    assert torch.equal(detail['logp0'], detail['logp1'])


def test_categorical_head_requires_captured_native_input():
    head, _, logits, targets = operands()
    with pytest.raises(ValueError, match='captured native input'):
        FiniteAnswerOps(compiled=False)(logits, head, targets)


def test_identical_inputs_do_not_attribute_bf16_output_discrepancy():
    head, hidden, logits, targets = operands()
    hidden[1] = hidden[0]
    with torch.no_grad():
        dense,detail=FiniteAnswerOps(compiled=False)(logits,head,targets,original_packed_hidden=hidden)
    assert torch.isfinite(dense).all()
    assert not (detail['packed_hidden']*(hidden[1:]-hidden[:1])).any()
    assert torch.equal(detail['logp0'],detail['logp1'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Original finite linear transpose is a GPU owner.')
def test_default_text_target_keeps_original_finite_seed():
    head,hidden,logits,targets=operands(device='cuda')
    targets.outcome_token_ids=None
    with torch.no_grad():
        expected=_answer_seed_rule(logits[0::2],logits[1::2],targets.labels,head.weight)
        _,detail=FiniteAnswerOps(compiled=False)(logits,head,targets)
    for actual,reference in zip((detail['packed_hidden'],detail['logp0'],detail['logp1'],
                                detail['allocated_logit_effect']),
                               (expected[0],expected[2],expected[3],expected[4])):
        assert torch.equal(actual,reference)


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


def test_outcome_readout_uses_head_during_its_forward_lifetime():
    # A small lifecycle regression, not a substitute for the real FSDP probe.
    # Make weights unavailable after the original Linear forward returns.
    from types import SimpleNamespace
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    head, hidden, _, _ = operands()
    hidden = hidden[:, None, :]
    expected = torch.nn.functional.linear(hidden[:, -1].float(), head.weight.float()).log_softmax(-1)
    runner = object.__new__(Qwen35DenseFiniteRunner)
    runner.model = SimpleNamespace(lm_head=head)

    def forward(input_ids, **kwargs):
        logits = head(hidden)
        head.weight = torch.nn.Parameter(torch.empty(head.weight.shape, device='meta'))
        return SimpleNamespace(logits=logits)

    runner.forward_prefix = forward
    actual = runner.read_outcomes(torch.zeros(2, 1, dtype=torch.long), [0, 1, 2])
    torch.testing.assert_close(actual, expected)
    assert head.weight.device.type == 'meta'
    assert not head._forward_pre_hooks


def test_outcome_readout_removes_hook_when_native_forward_fails():
    from types import SimpleNamespace
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    head, _, _, _ = operands()
    runner = object.__new__(Qwen35DenseFiniteRunner)
    runner.model = SimpleNamespace(lm_head=head)
    def forward(*args, **kwargs):
        raise RuntimeError('native failure')
    runner.forward_prefix = forward
    with pytest.raises(RuntimeError, match='native failure'):
        runner.read_outcomes(torch.zeros(1, 1, dtype=torch.long), [0, 1, 2])
    assert not head._forward_pre_hooks
