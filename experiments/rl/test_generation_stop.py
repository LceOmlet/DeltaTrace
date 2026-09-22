"""Real model metadata and original HF/VERL stop/mask behavior; no model double."""
import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation.stopping_criteria import EosTokenCriteria
from verl.utils.model import get_generation_config
from verl.utils.torch_functional import get_response_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def apply_owner_config_block(model_config, tokenizer, generation):
    # Execute the actual inserted owner block, not a duplicate implementation.
    source = ast.parse(textwrap.dedent(inspect.getsource(ActorRolloutRefWorker._build_model_optimizer)))
    blocks = [node for node in ast.walk(source) if isinstance(node, ast.If)
              and '_from_model_config' in ast.unparse(node.test)]
    assert len(blocks) == 1
    worker = SimpleNamespace(generation_config=generation, tokenizer=tokenizer)
    exec(compile(ast.Module(body=blocks, type_ignores=[]), '<actual-owner-config>', 'exec'),
         {'self': worker, 'actor_model_config': model_config})
    return worker.generation_config


def test_checkpoint_chat_end_stops_generation_and_keeps_terminator_in_loss_mask():
    path = os.environ['MODEL_PATH']
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    generation = get_generation_config(path)
    assert generation._from_model_config
    chat_end = tokenizer.eos_token_id
    old_stop = EosTokenCriteria(generation.eos_token_id)
    assert not old_stop(torch.tensor([[chat_end]]), scores=None).item()
    generation = apply_owner_config_block(config, tokenizer, generation)
    assert EosTokenCriteria(generation.eos_token_id)(torch.tensor([[chat_end]]), scores=None).item()
    ids = torch.tensor([[100, chat_end, tokenizer.pad_token_id, tokenizer.pad_token_id]])
    torch.testing.assert_close(get_response_mask(ids, generation.eos_token_id), torch.tensor([[1, 1, 0, 0]]))


@pytest.mark.parametrize('family,from_model', [('qwen3_5', False), ('qwen2', True)])
def test_explicit_configuration_and_other_families_remain_original(family, from_model):
    generation = GenerationConfig(eos_token_id=8, pad_token_id=9, temperature=.7,
                                  _from_model_config=from_model)
    before = generation.to_dict()
    apply_owner_config_block(SimpleNamespace(model_type=family), SimpleNamespace(eos_token_id=7), generation)
    assert generation.to_dict() == before
