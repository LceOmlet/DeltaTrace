"""Execute pinned VERL validation across native batches with unequal widths."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from agent_system.reward_manager.episode import EpisodeRewardManager


def original_validate():
    path = Path(os.environ['VERL_VALIDATION_BASELINE'])
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_validate')
    scope = dict(torch=torch, np=np, DataProto=DataProto)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_validate']


def run_validation(method, widths):
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, decode=lambda *a, **k: 'fixture')
    batches = []
    for step, width in enumerate(widths):
        rm = torch.zeros(2, width)
        rm[:, -1] = torch.tensor([1., .25]) if step == 0 else torch.tensor([0., .5])
        batches.append(DataProto.from_dict(tensors=dict(responses=torch.ones(2, width, dtype=torch.long),
            rm_scores=rm), non_tensors=dict(data_source=np.array(['sql']*2, dtype=object),
            tool_callings=np.array([1, 2]), traj_uid=np.array([f'{step}-0', f'{step}-1'], dtype=object))))
    iterator = iter(batches)
    rows = [dict(input_ids=torch.ones(2, 1, dtype=torch.long), attention_mask=torch.ones(2, 1),
        position_ids=torch.zeros(2, 1, dtype=torch.long), raw_prompt_ids=np.array([[1], [1]]),
        data_source=np.array(['sql']*2, dtype=object)) for _ in widths]
    calls = []
    trainer = SimpleNamespace(val_dataloader=rows, tokenizer=tokenizer,
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            val_kwargs=SimpleNamespace(n=1, do_sample=False))), reward_model=SimpleNamespace(enable=False)),
        actor_rollout_wg=None, val_envs=None,
        traj_collector=SimpleNamespace(multi_turn_loop=lambda **k: next(iterator)),
        val_reward_fn=EpisodeRewardManager(tokenizer, num_examine=0),
        _maybe_log_val_generations=lambda **kwargs: calls.append(kwargs))
    return method(trainer), calls


def test_equal_width_matches_unmodified_owner_exactly():
    assert run_validation(RayPPOTrainer._validate, [19, 19]) == run_validation(original_validate(), [19, 19])


def test_ragged_batches_match_owner_zero_padding_without_allocating_it():
    with pytest.raises(RuntimeError, match='Sizes of tensors'):
        run_validation(original_validate(), [19, 7])
    assert run_validation(RayPPOTrainer._validate, [19, 7]) == run_validation(original_validate(), [19, 19])
