"""Reuse the original native GRPO/DataProto/primitive contract fixtures."""
import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from counterfactual import reward_event_token_credit

def make_data(*, duplicate=False, source_d=None):
    outcomes = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
    reward = torch.zeros(8, 7)
    reward[:, -1] = outcomes
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0]]*8, dtype=torch.bool)
    # Use the actual pinned GRPO; there is no local group-normalization formula.
    group_advantage, _ = compute_grpo_outcome_advantage(
        token_level_rewards=reward, response_mask=mask,
        index=np.array(['native-test-group']*8), traj_index=np.arange(8),
        norm_adv_by_std_in_grpo=True)
    g = group_advantage[:, 0]
    artifacts = []
    for i in range(8):
        target = [False]*9
        if i != 0:
            target[7] = True
        policy = [True, False, True, True, False, True, False, True, False]
        artifacts.append(dict(prompt_ids=[8, 9], response_ids=list(range(100+9*i, 109+9*i)),
            policy_mask=policy, target_mask=target,
            retained_response_positions=[0, 1, 2, 3, 4, 7, -1]))
    objects = np.empty(8, dtype=object); objects[:] = artifacts
    responses = torch.tensor([[item['response_ids'][p] if p >= 0 else 0
                              for p in item['retained_response_positions']] for item in artifacts])
    data = DataProto.from_dict(tensors=dict(responses=responses, dt_grpo_multipliers=g,
        response_mask=mask), non_tensors=dict(traj_uid=np.array([f'native-{i}' for i in range(8)]),
        dt_direct_target_artifact=objects))
    if duplicate:
        data = data.select_idxs([0, 1, 2, 3, 4, 5, 6, 7, 1])
    return data

class PrimitiveContractWorker:
    """Native primitive oracle; deliberately not a model/DT replacement."""
    world_size = 2

    def __init__(self, d=None):
        self.calls = 0
        self.d = torch.tensor([.3, 0., -.1, .5, 0., .8, 0., 0., 0.]) if d is None else d

    def compute_dt_token_advantages(self, request):
        self.calls += 1
        width = request.batch['responses'].shape[-1]
        d = torch.zeros(len(request), 1, width)
        d[:, 0, :len(self.d)] = self.d
        credit = reward_event_token_credit(d, request.batch['dt_direct_reward'][:, None],
            request.batch['dt_future_event_mask'][:, None, :], request.batch['policy_mask'],
            self_target_mask=request.batch['target_mask'][:, None, :])
        return DataProto.from_dict(tensors=dict(dt_token_advantages=credit.advantages.float(),
            dt_q_estimates=credit.q_estimates.float(), dt_v_estimates=credit.v_estimates.float()))
