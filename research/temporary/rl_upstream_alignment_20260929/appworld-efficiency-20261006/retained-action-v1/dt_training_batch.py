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
            'rewards', 'episode_rewards', 'traj_uid', 'env_step', 'active_masks',
            'appworld_num_tests')
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


def compute_training_credit(data, worker_group, *, eos_token_id, pad_token_id, source_indices=None):
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

    source, inverse = prepare_training_credit(data)
    shape = source.batch['responses'].shape
    result = {key: torch.zeros(shape, dtype=torch.float32) for key in CREDIT_KEYS}
    active = torch.tensor([bool(value) for value in source.non_tensor_batch['active_masks']])
    policy_lengths = source.batch['attention_mask'][:, -shape[1]:].sum(-1).cpu()
    nonzero = (source.batch['dt_complete_return'] != 0) & active & (policy_lengths > 0)
    if source_indices is not None:
        # Truncation selects training tokens, not environment reward events.
        # Complete returns above still include every executed future row.
        eligible = torch.zeros(len(source), dtype=torch.bool)
        eligible[inverse[source_indices]] = True
        nonzero &= eligible
    indices = nonzero.nonzero().flatten()
    if len(indices):
        # Same number of nonzero B4 calls on every FSDP rank. Zero-return
        # responses require no model call and remain zero at their identities.
        # The owner's native padding duplicates at most world_size*4-1 rows.
        lengths = source.batch['attention_mask'].sum(-1).cpu()
        indices = indices[torch.argsort(lengths[indices], stable=True)]
        groups = [indices]
        if 'appworld_num_tests' in source.non_tensor_batch:
            counts = torch.tensor([int(n) for n in source.non_tensor_batch['appworld_num_tests']])
            groups = [indices[counts[indices] == count] for count in counts[indices].unique(sorted=True)]
        for group in groups:
            # A collective batch uses one native reward alphabet on all ranks.
            # This changes packaging only; every actual response is traced once.
            requests = source.select_idxs(group)
            requests.meta_info.update(eos_token_id=int(eos_token_id), pad_token_id=int(pad_token_id))
            requests, padding = pad_dataproto_to_divisor(requests, worker_group.world_size * 4)
            # Use the trainer owner's partitioner before its contiguous DP split.
            # Splitting globally sorted rows directly puts all long contexts on
            # one rank, while every FSDP microbatch waits for that rank.
            partitions = get_seqlen_balanced_partitions(
                requests.batch['attention_mask'].sum(-1).tolist(),
                k_partitions=worker_group.world_size, equal_size=True)
            order = torch.tensor([index for partition in partitions for index in partition])
            requests.reorder(order)
            values = worker_group.compute_dt_token_advantages(requests)
            # Restore the original padding positions before native unpadding,
            # then scatter to the unchanged factual response identities.
            values.reorder(torch.argsort(order))
            values = unpad_dataproto(values, padding)
            for key in CREDIT_KEYS:
                result[key][group] = values.batch[key].cpu()
    return DataProto.from_dict(tensors={key: value[inverse] for key, value in result.items()})
