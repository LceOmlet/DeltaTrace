"""Pinned VERL owner interfaces, with explicit model doubles for transport.

These tests prove index/shape/default-path behavior, not Qwen quality or speed.
"""
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.rollout.hf_rollout import HFRollout


class HeadOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 32)
        self.lengths = []

    def forward(self, input_ids, attention_mask, position_ids, logits_to_keep, **kwargs):
        self.lengths.append(input_ids.shape[-1])
        logits = self.weight[input_ids[:, -logits_to_keep:]]
        return SimpleNamespace(logits=logits)


def test_actor_shared_padding_keeps_original_response_logprobs_and_gradient(monkeypatch):
    import verl.utils.torch_functional as functional
    # Exercise the owner's existing CPU logprob branch in this CPU-only test;
    # the installed optional FlashAttention cross entropy is CUDA-only.
    monkeypatch.setattr(functional, 'FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE', False)
    ids = torch.zeros((4, 32768), dtype=torch.long)
    ids[:, -8:] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])
    attention = torch.zeros_like(ids);attention[:, -8:] = 1
    attention[0, -9] = 1  # Only shared padding may be removed.
    positions = (attention.cumsum(-1)-1).clamp_min(0)
    batch = dict(input_ids=ids, attention_mask=attention, position_ids=positions, responses=ids[:, -4:])
    model = HeadOnlyModel()
    actor = SimpleNamespace(actor_module=model, device_name='cpu', use_remove_padding=False,
                            use_fused_kernels=False)
    gradients, probabilities = [], []
    for flag in ['0', '1']:
        monkeypatch.setenv('VERL_TRIM_SHARED_PADDING', flag)
        _, lp = DataParallelPPOActor._forward_micro_batch(actor, batch, 1.0, False)
        probabilities.append(lp.detach())
        lp.sum().backward();gradients.append(model.weight.grad.clone());model.weight.grad = None
    assert model.lengths == [32768, 9]
    torch.testing.assert_close(*probabilities)
    torch.testing.assert_close(*gradients)
    assert batch['input_ids'].shape == (4, 32768)


class GenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def generate(self, *, input_ids, attention_mask, position_ids, generation_config, **kwargs):
        self.inputs.append((input_ids.clone(), attention_mask.clone(), position_ids.clone()))
        ids = input_ids.repeat_interleave(generation_config.num_return_sequences, dim=0)
        suffix = torch.tensor([[5, 6, 7]], device=ids.device).expand(ids.shape[0], -1)
        return SimpleNamespace(sequences=torch.cat((ids, suffix), dim=-1))


@pytest.mark.parametrize('copies', [1, 2])
def test_hf_generation_removes_compute_padding_and_restores_original_dataproto(monkeypatch, copies):
    import verl.workers.rollout.hf_rollout as owner
    monkeypatch.setattr(owner, 'get_device_name', lambda: 'cpu')
    monkeypatch.setattr(owner, 'get_device_id', lambda: 0)
    monkeypatch.setattr(owner, 'get_torch_device', lambda: SimpleNamespace(empty_cache=lambda: None))
    ids = torch.zeros((2, 32256), dtype=torch.long)
    ids[:, -3:] = torch.tensor([1, 2, 3])
    mask = ids.ne(0).long();pos = (mask.cumsum(-1)-1).clamp_min(0)
    prompts = DataProto(batch=TensorDict(dict(input_ids=ids, attention_mask=mask, position_ids=pos), batch_size=[2]),
                       meta_info={'eos_token_id':7, 'pad_token_id':0})
    config = OmegaConf.create(dict(do_sample=True, temperature=1., response_length=512, top_p=1., top_k=0, n=copies))
    model = GenerationModel();rollout = HFRollout(model, config)
    outputs = []
    for flag in ['0', '1']:
        monkeypatch.setenv('VERL_TRIM_SHARED_PADDING', flag)
        outputs.append(rollout._generate_minibatch(prompts))
    assert [value[0].shape[-1] for value in model.inputs] == [32256, 3]
    assert outputs[1].batch['input_ids'].shape == (2*copies, 32768)
    for name in outputs[0].batch.keys():
        torch.testing.assert_close(outputs[0].batch[name], outputs[1].batch[name])
