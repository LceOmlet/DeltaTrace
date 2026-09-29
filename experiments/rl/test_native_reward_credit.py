"""DT consumes the actual pinned VERL reward path, including native penalties."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import adjust_batch
from agent_system.reward_manager.episode import EpisodeRewardManager
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty
from counterfactual import return_credit_for_episode
from deltatrace_rollout import DeltaTraceRolloutProducer
from reward_readout import RewardAlphabet


@pytest.mark.parametrize('task,rewards,expected', [
    ('Sokoban', [-.1, 10.9], [10.7, 10.9]),
    ('Webshop', [0., 10.], [9.9, 10.]),
    ('AppWorld', [0., 10.], [9.9, 10.]),
    ('Webshop', [0., 0.], [-.1, 0.]),
])
def test_native_penalty_then_dt_keeps_training_order(task, rewards, expected):
    rows = [dict(
        traj_uid='episode', env_step=i, rewards=reward, active_masks=True,
        is_action_valid=np.array([i != 0]), data_source=task,
        prompts=torch.tensor([40, 41]), responses=torch.tensor([7, 8, 0]),
        input_ids=torch.tensor([40, 41, 7, 8, 0]),
        attention_mask=torch.tensor([1, 1, 1, 1, 0]),
    ) for i, reward in enumerate(rewards)]
    collector = TrajectoryCollector(SimpleNamespace(algorithm=SimpleNamespace(adv_estimator='deltatrace')), None)
    data = collector.gather_rollout_data(
        [rows], np.array([sum(rewards)]), np.array([2]), {}, np.array(['episode']), np.array([0]),
    )
    cfg = OmegaConf.create(dict(
        trainer=dict(n_gpus_per_node=1, nnodes=1),
        algorithm=dict(use_kl_in_reward=False), actor_rollout_ref=dict(
            actor=dict(use_kl_loss=False, ppo_micro_batch_size_per_gpu=4),
            rollout=dict(log_prob_micro_batch_size_per_gpu=4))))
    # The actual owner duplicates two rows to fill B4. Test the subsequently
    # reordered batch, rather than assuming the collector order survives.
    data = adjust_batch(cfg, data).select_idxs([3, 0, 2, 1])
    tokenizer = SimpleNamespace(decode=lambda *a, **kw: '')
    data.batch['token_level_scores'] = EpisodeRewardManager(tokenizer, num_examine=0)(data)
    data, _ = apply_invalid_action_penalty(data, invalid_action_penalty_coef=.1)
    data.batch['token_level_rewards'] = data.batch['token_level_scores']
    producer = DeltaTraceRolloutProducer.__new__(DeltaTraceRolloutProducer)
    calls = []

    def trace_fixture(self, episodes, returns):
        calls.append(episodes)
        return [return_credit_for_episode(episode, [torch.tensor([.2, -.3, 0.]) for _ in episode])
                for episode in episodes]

    producer.attribute_episodes = MethodType(trace_fixture, producer)
    output = producer.attribute_training_batch(data)
    assert len(calls) == 1 and len(calls[0]) == 1 and len(calls[0][0]) == 2
    alphabet = RewardAlphabet.for_task(task, 15, .1)
    for i, step in enumerate(data.non_tensor_batch['env_step']):
        q = output.batch['dt_q_estimates'][i]
        torch.testing.assert_close(q, torch.tensor([expected[step], expected[step], 0.]))
        alphabet.observed_index(float(q[0]))
        torch.testing.assert_close(output.batch['dt_token_advantages'][i],
                                   q * -torch.expm1(-torch.tensor([.2, -.3, 0.])))
    # Native penalty affects this response once; future invalid penalties are
    # not invented or accumulated by the adapter, and raw rewards are intact.
    assert [row['rewards'] for row in rows] == rewards


def test_official_loss_and_reward_functions_are_unchanged():
    import ast
    import os
    from pathlib import Path
    if 'DT_PRISTINE_VERL' not in os.environ:
        pytest.skip('requires the recorded unmodified pinned author archive')
    current = Path(os.environ['VERL_ROOT'])
    original = Path(os.environ['DT_PRISTINE_VERL'])
    for relative, names in [
        ('verl/trainer/ppo/ray_trainer.py', ('apply_invalid_action_penalty',)),
        ('verl/trainer/ppo/core_algos.py', ('compute_policy_loss', 'agg_loss')),
        ('verl/workers/actor/dp_actor.py', ('update_policy', '_optimizer_step')),
    ]:
        left, right = [ast.parse((root / relative).read_text()) for root in (current, original)]
        for name in names:
            a, b = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
                    for tree in (left, right)]
            assert ast.dump(a) == ast.dump(b), (relative, name)
