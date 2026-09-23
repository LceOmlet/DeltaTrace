"""Reproduce PPO-before-DT root initialization with native PEFT/Qwen/FSDP2.

This checks the owner call boundary, not a model-quality or kernel tolerance.
"""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from deltatrace_rollout import _Qwen35CausalOwnerView


@pytest.mark.parametrize('actor_first', [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires FSDP2 GPU')
def test_owner_and_actor_share_official_root(tmp_path, actor_first):
    created = not torch.distributed.is_initialized()
    if created:
        torch.distributed.init_process_group('nccl', init_method='file://'+str(tmp_path/'rdzv'), rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            layer_types=['full_attention']*4, pad_token_id=0,
            rope_parameters=dict(rope_type='default', rope_theta=10000., partial_rotary_factor=1., mrope_section=[4,4,8]))
        config._attn_implementation = 'sdpa'
        parent = Qwen3_5Config(text_config=config.to_dict())
        model = get_peft_model(Qwen3_5ForCausalLM(config).cuda().bfloat16(),
            LoraConfig(r=1, lora_alpha=2, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'),
            autocast_adapter_dtype=False)
        base = model.get_base_model()
        mesh = init_device_mesh('cuda', (1,))
        fully_shard(base.model.embed_tokens, mesh=mesh)
        for layer in base.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        ids = torch.randint(1, 128, (2, 33), device='cuda')
        model.eval()
        if actor_first:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
        view = _Qwen35CausalOwnerView(model, base.model, base.lm_head, parent)
        for _ in range(2):
            with torch.no_grad():
                owner = view(input_ids=ids, use_cache=False, logits_to_keep=1).logits
                view.release_owner_params()
                actor = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
            torch.testing.assert_close(owner, actor, rtol=0, atol=0)
            model.train()
            loss = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float().square().mean()
            loss.backward()
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
            model.zero_grad(set_to_none=True)
            model.eval()
    finally:
        if created:
            torch.distributed.destroy_process_group()
