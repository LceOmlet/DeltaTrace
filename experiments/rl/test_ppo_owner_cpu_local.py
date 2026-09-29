"""Small CPU integration against the actual pinned VERL actor and HF model.

No PPO oracle is reimplemented. This checks the default path of the local
head-retention patch, not GPU/FA/FLA numerical acceptance or the 9B model.
"""
import ast
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
assert not torch.cuda.is_available(), 'This is a CPU-only local test'
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl import DataProto
from verl.workers.actor import dp_actor as owner

from patch_verl_agent2 import ACTOR_FORWARD_OLD, ACTOR_FORWARD_NEW, patch_actor_response_head


def patched_actor():
    source = Path(owner.__file__).read_text(encoding='utf-8')
    assert source.count(ACTOR_FORWARD_OLD) == 1
    changed = patch_actor_response_head(source.replace(ACTOR_FORWARD_OLD, ACTOR_FORWARD_NEW, 1))
    before = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
    after = next(n for n in ast.parse(changed).body if isinstance(n, ast.ClassDef) and n.name == before.name)
    old_methods = {n.name: ast.dump(n) for n in before.body if isinstance(n, ast.FunctionDef)}
    new_methods = {n.name: ast.dump(n) for n in after.body if isinstance(n, ast.FunctionDef)}
    assert {k for k in old_methods if old_methods[k] != new_methods[k]} == {'_forward_micro_batch'}
    namespace = dict(vars(owner))
    exec(compile(ast.Module(body=[after], type_ignores=[]), owner.__file__, 'exec'), namespace)
    return namespace['DataParallelPPOActor']


def test_cpu_head_retention_preserves_native_actor_updates(monkeypatch):
    monkeypatch.setenv('VERL_TRIM_RESPONSE_HEAD', '0')
    from verl.utils.debug import performance
    # Only the GPU-only logging probe is unavailable on CPU. Do not fake its
    # memory figures; the external launcher measures real RSS/physical VRAM.
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: ('CPU fixture: unavailable',) * 4)
    root = Path(owner.__file__).parents[3]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'verl/trainer/config')):
        cfg = compose(config_name='ppo_trainer').actor_rollout_ref.actor
    # Small CPU fixture and no compiler; retain native loss/entropy/dual clip.
    cfg.ppo_mini_batch_size = 4
    cfg.ppo_micro_batch_size_per_gpu = 4
    cfg.use_torch_compile = False
    cfg.use_kl_loss = True
    torch.manual_seed(719)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=32))
    other = deepcopy(model)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    controls = [owner.DataParallelPPOActor(cfg, model, torch.optim.AdamW(model.parameters(), lr=1e-5)),
                patched_actor()(deepcopy(cfg), other, torch.optim.AdamW(other.parameters(), lr=1e-5))]
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
    batch = DataProto.from_single_dict(dict(input_ids=ids, attention_mask=mask,
        position_ids=torch.arange(8).expand(4, -1), responses=ids[:, -4:]))
    batch.meta_info.update(temperature=1., micro_batch_size=4, use_dynamic_bsz=False)
    old, _ = controls[0].compute_log_prob(deepcopy(batch), calculate_entropy=True)
    new, _ = controls[1].compute_log_prob(deepcopy(batch), calculate_entropy=True)
    assert torch.equal(old, new)
    batch.batch['old_log_probs'] = old
    batch.batch['ref_log_prob'] = old + .05
    batch.batch['advantages'] = torch.tensor([[.2, -.5, 1., 0.]] * 4)
    for _ in range(2):
        results = [actor.update_policy(deepcopy(batch)) for actor in controls]
        assert results[0] == results[1]
        for a, b in zip(model.parameters(), other.parameters()):
            assert torch.equal(a, b)
    assert any(not torch.equal(value, initial[name]) for name, value in model.named_parameters())
