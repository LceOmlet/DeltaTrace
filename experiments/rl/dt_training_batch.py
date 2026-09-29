"""Prepare complete DT returns before VERL distributes response rows.

VERL owns chunking, transport, padding and collection. This credit boundary
retains factual identities across native balancing/duplication and keeps the
complete future return when rows of one trajectory go to different ranks.
"""
import torch

from counterfactual import episode_returns


CREDIT_KEYS = ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates')


def prepare_training_credit(data):
    from agent_system.multi_turn_rollout.utils import to_list_of_dict

    keys = ('input_ids', 'attention_mask', 'responses', 'token_level_rewards',
            'rewards', 'episode_rewards', 'traj_uid', 'env_step', 'active_masks')
    source = data.select(
        batch_keys=[key for key in keys if key in data.batch],
        non_tensor_batch_keys=[key for key in keys if key in data.non_tensor_batch],
    )
    rows = to_list_of_dict(source)
    grouped, unique, first_indices, inverse = {}, {}, [], []
    for index, row in enumerate(rows):
        identity = (str(row['traj_uid']), int(row['env_step']))
        if identity not in unique:
            unique[identity] = len(first_indices)
            first_indices.append(index)
            row['dt_reward_adjustment'] = (
                float(row['token_level_rewards'].double().sum()) - float(row['episode_rewards']))
            grouped.setdefault(identity[0], {})[identity[1]] = row
        inverse.append(unique[identity])
    returns = torch.zeros(len(first_indices), dtype=torch.float64)
    for steps in grouped.values():
        episode = [steps[key] for key in sorted(steps)]
        for row, value in zip(episode, episode_returns(episode)):
            returns[unique[(str(row['traj_uid']), int(row['env_step']))]] = value
    source = source.select_idxs(first_indices)
    source.batch['dt_complete_return'] = returns
    return source, torch.tensor(inverse, dtype=torch.long)


def compute_training_credit(data, worker_group, *, eos_token_id, pad_token_id):
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    source, inverse = prepare_training_credit(data)
    shape = source.batch['responses'].shape
    result = {key: torch.zeros(shape, dtype=torch.float32) for key in CREDIT_KEYS}
    active = torch.tensor([bool(value) for value in source.non_tensor_batch['active_masks']])
    policy_lengths = source.batch['attention_mask'][:, -shape[1]:].sum(-1).cpu()
    nonzero = (source.batch['dt_complete_return'] != 0) & active & (policy_lengths > 0)
    indices = nonzero.nonzero().flatten()
    if len(indices):
        # Same number of nonzero B4 calls on every FSDP rank. Zero-return
        # responses require no model call and remain zero at their identities.
        # The owner's native padding duplicates at most world_size*4-1 rows.
        lengths = source.batch['attention_mask'].sum(-1).cpu()
        indices = indices[torch.argsort(lengths[indices], stable=True)]
        requests = source.select_idxs(indices)
        requests.meta_info.update(eos_token_id=int(eos_token_id), pad_token_id=int(pad_token_id))
        requests, padding = pad_dataproto_to_divisor(requests, worker_group.world_size * 4)
        values = unpad_dataproto(worker_group.compute_dt_token_advantages(requests), padding)
        for key in CREDIT_KEYS:
            result[key][indices] = values.batch[key].cpu()
    return DataProto.from_dict(tensors={key: value[inverse] for key, value in result.items()})
