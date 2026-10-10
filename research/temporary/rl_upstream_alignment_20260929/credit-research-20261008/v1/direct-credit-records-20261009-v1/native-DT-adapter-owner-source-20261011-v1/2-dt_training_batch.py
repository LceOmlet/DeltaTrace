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
    nonzero_before = int(nonzero.sum())
    retained_sources = len(source)
    if source_indices is not None:
        # Truncation selects training tokens, not environment reward events.
        # Complete returns above still include every executed future row.
        eligible = torch.zeros(len(source), dtype=torch.bool)
        eligible[inverse[source_indices]] = True
        retained_sources = int(eligible.sum())
        nonzero &= eligible
    indices = nonzero.nonzero().flatten()
    print(f'[DT source workload] response_rows={len(data)} unique_rows={len(source)} '
          f'retained_sources={retained_sources} nonzero_before={nonzero_before} '
          f'nonzero_after={len(indices)} skipped_nonzero={nonzero_before-len(indices)}', flush=True)
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


def compute_direct_target_credit(data, worker_group, *, eos_token_id, pad_token_id):
    """One joint actual-action target per original native trajectory.

    Owners supply exact full-history IDs/masks and their retained training
    positions. VERL supplies rewards, padding, DP partitions and RPC. This
    boundary neither parses actions nor reconstructs/truncates a trajectory.
    """
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.utils.torch_functional import postprocess_data, pad_2d_list_to_length
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
    import numpy as np

    artifacts = data.non_tensor_batch['dt_direct_target_artifact']
    identities = data.non_tensor_batch['traj_uid']
    first, unique, inverse = [], {}, []
    for index, identity in enumerate(identities):
        identity = str(identity)
        if identity not in unique:
            unique[identity] = len(first)
            first.append(index)
        inverse.append(unique[identity])
    selected = [artifacts[index] for index in first]
    rewards = data.batch['token_level_rewards'].double().sum(-1).cpu()[first]
    destination = {key: torch.zeros_like(data.batch['responses'], dtype=torch.float32)
                   for key in CREDIT_KEYS}
    if not selected:
        return DataProto.from_dict(tensors=destination)

    prompt_width = max(len(item['prompt_ids']) for item in selected)
    prompts, prompt_masks = zip(*(postprocess_data(
        torch.tensor([item['prompt_ids']], dtype=torch.long),
        torch.ones(1, len(item['prompt_ids']), dtype=torch.long), prompt_width,
        int(pad_token_id), left_pad=True, truncation='error') for item in selected))
    prompts, prompt_masks = torch.cat(prompts), torch.cat(prompt_masks)
    responses = pad_2d_list_to_length([item['response_ids'] for item in selected], int(pad_token_id))
    width = responses.shape[-1]
    response_attention = torch.arange(width)[None, :] < torch.tensor(
        [len(item['response_ids']) for item in selected])[:, None]
    policy = pad_2d_list_to_length([item['policy_mask'] for item in selected], 0).bool()
    target = pad_2d_list_to_length([item['target_mask'] for item in selected], 0).bool()
    if policy.shape != responses.shape or target.shape != responses.shape:
        raise ValueError('native direct target masks must retain the full response IDs')
    if bool((target & ~policy).any()):
        raise ValueError('native action targets must be policy tokens')
    attention = torch.cat((prompt_masks, response_attention.long()), -1)
    requests = DataProto.from_dict(tensors=dict(
        input_ids=torch.cat((prompts, responses), -1), prompts=prompts,
        responses=responses, attention_mask=attention,
        position_ids=compute_position_id_with_mask(attention),
        policy_mask=policy, target_mask=target, dt_direct_reward=rewards),
        non_tensors=dict(traj_uid=np.array([str(identities[index]) for index in first], dtype=object)))
    requests.meta_info.update(eos_token_id=int(eos_token_id), pad_token_id=int(pad_token_id),
                              dt_target_semantics='native_joint_action_target')
    result = {key: torch.zeros_like(responses, dtype=torch.float32) for key in CREDIT_KEYS}
    positions = torch.arange(width)[None, :]
    last_target = torch.where(target, positions, -1).max(-1).values
    preceding_source = (policy & ~target & (positions < last_target[:, None])).any(-1)
    trace_required = (rewards != 0) & target.any(-1) & preceding_source
    # Known self/empty-target/causal-zero cases use the SAME Q/V primitive on
    # the driver. Only actual DT requests enter the collective worker call;
    # every rank therefore receives the same number of B4 owner invocations.
    from counterfactual import reward_event_token_credit
    analytic = reward_event_token_credit(
        torch.zeros((len(first), 1, width), dtype=torch.float32), rewards[:, None],
        torch.ones((len(first), 1, width), dtype=torch.bool), policy,
        self_target_mask=target[:, None, :])
    result.update(dt_token_advantages=analytic.advantages.float(),
                  dt_q_estimates=analytic.q_estimates.float(),
                  dt_v_estimates=analytic.v_estimates.float())
    indices = trace_required.nonzero().flatten()
    print(f'[DT direct target workload] trajectories={len(first)} '
          f'nonzero_requests={len(indices)} target_tokens={int(target.sum())} '
          f'source_tokens={int((policy & ~target).sum())}', flush=True)
    if len(indices):
        indices = indices[torch.argsort(attention.sum(-1)[indices], stable=True)]
        active = requests.select_idxs(indices)
        active, padding = pad_dataproto_to_divisor(active, worker_group.world_size * 4)
        partitions = get_seqlen_balanced_partitions(
            active.batch['attention_mask'].sum(-1).tolist(),
            k_partitions=worker_group.world_size, equal_size=True)
        order = torch.tensor([index for partition in partitions for index in partition])
        active.reorder(order)
        values = worker_group.compute_dt_token_advantages(active)
        values.reorder(torch.argsort(order))
        values = unpad_dataproto(values, padding)
        for key in CREDIT_KEYS:
            result[key][indices] = values.batch[key].cpu()
    for row, source_index in enumerate(inverse):
        item = artifacts[row]
        positions = torch.as_tensor(item['retained_response_positions'], dtype=torch.long)
        if positions.shape != data.batch['responses'][row].shape:
            raise ValueError('native retained positions must align with original training responses')
        kept = positions >= 0
        if bool(kept.any()):
            if bool((positions[kept] >= len(selected[source_index]['response_ids'])).any()):
                raise ValueError('native retained position is outside the full trajectory')
            original = torch.as_tensor(selected[source_index]['response_ids'], dtype=torch.long)
            if not torch.equal(data.batch['responses'][row, kept].cpu(), original[positions[kept]]):
                raise ValueError('native retained token identity differs from the original full trajectory')
            for key in CREDIT_KEYS:
                destination[key][row, kept] = result[key][source_index, positions[kept]]
    return DataProto.from_dict(tensors=destination)
