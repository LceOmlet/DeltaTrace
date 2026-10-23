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
    # The trainer supplies the unchanged official GRPO result before deduplication.
    # This replaces the return multiplier; raw environment rewards remain untouched.
    import os
    multiplier_key = ('dt_reward_multipliers' if os.environ.get('DT_CREDIT_MODE') == 'reward_attribution_then_group_v12'
                      else 'dt_grpo_multipliers')
    rewards = data.batch[multiplier_key].double().cpu()[first]
    destination = {key: torch.zeros_like(data.batch['responses'], dtype=torch.float32)
                   for key in CREDIT_KEYS}
    if not selected:
        return DataProto.from_dict(tensors=destination)

    import os
    if os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v5', 'reward_target_logprob_exponential_v10', 'reward_target_exponential_v11', 'reward_attribution_then_group_v12') and any(
            'dt_reward_target_events' in item for item in selected):
        return _compute_multi_target_credit(data, selected, first, inverse, rewards,
            worker_group, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

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
    future = pad_2d_list_to_length([
        item.get('future_policy_mask', [True] * len(item['response_ids']))
        for item in selected], 0).bool()
    if policy.shape != responses.shape or target.shape != responses.shape:
        raise ValueError('native direct target masks must retain the full response IDs')
    if bool((target & ~policy).any()):
        raise ValueError('native action targets must be policy tokens')
    attention = torch.cat((prompt_masks, response_attention.long()), -1)
    requests = DataProto.from_dict(tensors=dict(
        input_ids=torch.cat((prompts, responses), -1), prompts=prompts,
        responses=responses, attention_mask=attention,
        position_ids=compute_position_id_with_mask(attention),
        policy_mask=policy, target_mask=target, dt_future_event_mask=future,
        dt_direct_reward=rewards),
        non_tensors=dict(traj_uid=np.array([str(identities[index]) for index in first], dtype=object)))
    import os
    weighted = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'
    if weighted:
        spans = np.empty(len(selected), dtype=object)
        for i,item in enumerate(selected):
            spans[i] = [(span['target_offsets'],span['credit_fraction'])
                        for span in item['reward_target_spans'] if span['target_offsets']]
        requests.non_tensor_batch['dt_target_span_credits'] = spans
    requests.meta_info.update(eos_token_id=int(eos_token_id), pad_token_id=int(pad_token_id),
                              dt_target_semantics='native_joint_action_target')
    result = {key: torch.zeros_like(responses, dtype=torch.float32) for key in CREDIT_KEYS}
    positions = torch.arange(width)[None, :]
    last_target = torch.where(target, positions, -1).max(-1).values
    preceding_source = (policy & future & ~target & (positions < last_target[:, None])).any(-1)
    trace_required = (rewards != 0) & target.any(-1) & preceding_source
    # Known self/empty-target/causal-zero cases use the SAME Q/V primitive on
    # the driver. Only actual DT requests enter the collective worker call;
    # every rank therefore receives the same number of B4 owner invocations.
    from counterfactual import reward_event_token_credit
    self_boundary = target[:, None, :]
    if os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v11', 'reward_attribution_then_group_v12'):
        empty_target = ~target.any(-1)
        # A flattened event may be empty while another event of the same
        # rollout has a target. Only a genuinely target-free rollout falls back.
        empty_target &= torch.tensor([not item.get('dt_rollout_has_target', False)
                                      for item in selected], dtype=torch.bool)
        self_boundary = (target | (empty_target[:, None] & policy))[:, None, :]
        print(f'[DT empty-target broadcast] trajectories={len(first)} '
              f'empty_targets={int(empty_target.sum())} '
              f'nonzero_g_fallback={int((empty_target & (rewards != 0)).sum())}', flush=True)
    analytic = reward_event_token_credit(
        torch.zeros((len(first), 1, width), dtype=torch.float32), rewards[:, None],
        future[:, None, :], policy,
        self_target_mask=self_boundary, linear_attribution=weighted)
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
    return _restore_direct_target_credit(data, selected, inverse, result, destination)


def _restore_direct_target_credit(data, selected, inverse, result, destination=None):
    """Restore the owner's exact retained positions for both event packings."""
    from verl import DataProto
    if destination is None:
        destination = {key: torch.zeros_like(data.batch['responses'], dtype=torch.float32)
                       for key in CREDIT_KEYS}
    artifacts = data.non_tensor_batch['dt_direct_target_artifact']
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


def _compute_multi_target_credit(data, selected, first, inverse, rewards, worker_group,
                                 *, eos_token_id, pad_token_id):
    """Pack explicit credited target events before the original collective DP.

    dt_reward_target_events contains unique event_id, original target_offsets,
    detached signed credit b_k, and an explicit future_policy_mask. The caller
    owns these credits and reward identities. No group statistics, b/B, d,
    or u normalization is reconstructed here. Multiple payload pieces of one
    event belong in that event's single target_offsets union.

    Flatten BEFORE dispatch so all FSDP ranks receive the same B4 call count.
    The existing one-event readout exponentiates each native DT vector, then
    only the returned Q/V/A are summed at their original trajectory slots.
    """
    import math
    import os
    import numpy as np
    from verl import DataProto
    from verl.utils.torch_functional import pad_2d_list_to_length

    flattened, coefficients, identities, owners, event_ids = [], [], [], [], []
    for source_index, item in enumerate(selected):
        events = item.get('dt_reward_target_events')
        if events is None:
            events = [dict(event_id=0, target_offsets=[i for i,t in enumerate(item['target_mask']) if t],
                           credit=float(rewards[source_index]), reward=float(rewards[source_index]),
                           future_policy_mask=[True] * len(item['response_ids']))]
        if not events:
            raise ValueError('Explicit reward target events must retain their settlement identities')
        seen = set()
        for event in events:
            event_id = event['event_id']
            if not isinstance(event_id, (int, str)):
                raise ValueError('Reward target event identity must be an original integer or string')
            identity = (type(event_id).__name__, event_id)
            if identity in seen:
                raise ValueError('One reward settlement cannot be repeated as multiple credit events')
            seen.add(identity)
            # A full signed component is required. Repeating rollout g for
            # every span would silently change the agreed credit construction.
            coefficient_key = 'reward' if os.environ.get('DT_CREDIT_MODE') == 'reward_attribution_then_group_v12' else 'credit'
            if coefficient_key not in event:
                raise ValueError('Each target event requires its own already computed stable credit')
            coefficient = float(event['reward'] if os.environ.get('DT_CREDIT_MODE') == 'reward_attribution_then_group_v12' else event['credit'])
            if not math.isfinite(coefficient):
                raise ValueError('Target event credit must be finite')
            width = len(item['response_ids'])
            target = [False] * width
            offsets = event['target_offsets']
            if len(set(offsets)) != len(offsets):
                raise ValueError('A target event must select each original token position once')
            for position in offsets:
                if not 0 <= position < width or not item['policy_mask'][position]:
                    raise ValueError('Target event offsets must refer to original policy tokens')
                target[position] = True
            future = torch.as_tensor(event['future_policy_mask'])
            if future.shape != (width,) or future.dtype != torch.bool:
                raise ValueError('Target event future mask must be bool at original full suffix slots')
            artifact = dict(item, target_mask=target, future_policy_mask=future.tolist(),
                            retained_response_positions=list(range(width)))
            import os
            if os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v11', 'reward_attribution_then_group_v12'):
                artifact['dt_rollout_has_target'] = any(e['target_offsets'] for e in events)
            artifact.pop('dt_reward_target_events', None)
            flattened.append(artifact)
            coefficients.append(coefficient)
            original_uid = str(data.non_tensor_batch['traj_uid'][first[source_index]])
            identities.append(str((original_uid, *identity)))
            owners.append(source_index);event_ids.append(event_id)
    responses = pad_2d_list_to_length([item['response_ids'] for item in flattened], int(pad_token_id))
    width = responses.shape[-1]
    for item in flattened:
        item['retained_response_positions'] += [-1] * (width-len(item['response_ids']))
    objects = np.empty(len(flattened), dtype=object);objects[:] = flattened
    event_data = DataProto.from_dict(tensors=dict(responses=responses,
        **{('dt_reward_multipliers' if os.environ.get('DT_CREDIT_MODE') == 'reward_attribution_then_group_v12' else 'dt_grpo_multipliers'):
           torch.tensor(coefficients, dtype=torch.float64)}),
        non_tensors=dict(traj_uid=np.array(identities,dtype=object),dt_direct_target_artifact=objects))
    # Same owner dedup/pad/balance/RPC/scatter implementation as the K=1 path.
    per_event = compute_direct_target_credit(event_data, worker_group,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    owner_index = torch.tensor(owners, dtype=torch.long)
    result = {key: torch.zeros((len(selected), width),dtype=torch.float32).index_add_(
        0, owner_index, per_event.batch[key].cpu()) for key in CREDIT_KEYS}
    print(f'[DT multi-target credit] trajectories={len(selected)} events={len(flattened)} '
          f'event_ids={event_ids} transform=per_event_exponential_then_sum',flush=True)
    return _restore_direct_target_credit(data, selected, inverse, result)


def prepare_raw_outcome_credit(data):
    """Use the pinned GRPO owner's outcome reduction, without group-normalizing it."""
    # Same native outcome: environment scores after the owner's reward adjustments.
    data.batch['dt_reward_multipliers'] = data.batch['token_level_rewards'].sum(-1).detach()
    return data


@torch.no_grad()
def apply_attributed_group_advantage(data, epsilon=1e-6):
    """One retained-policy reward table, one pooled mean/std per original uid.

    DT's raw A0 is the shaped reward r*u (or r on self/target-free tokens).
    Q=Q0/s, V=(V0+mu)/s preserve A=Q-V under this group affine transform.
    Observation/padding are excluded; no second batch variance scaling follows.
    """
    mask = data.batch['response_mask'].bool()
    raw = data.batch['dt_token_advantages'].double()
    q0 = data.batch['dt_q_estimates'].double()
    v0 = data.batch['dt_v_estimates'].double()
    groups = {}
    for row, uid in enumerate(data.non_tensor_batch['uid']):
        groups.setdefault(uid, []).append(row)
    result = {name: torch.zeros_like(raw) for name in CREDIT_KEYS}
    statistics = []
    for uid, rows in groups.items():
        index = torch.as_tensor(rows, device=raw.device)
        active = mask[index]
        selected = raw[index][active]
        if selected.numel() == 0:
            continue
        # Pinned GRPO uses these PyTorch reductions; here they see the token table.
        mean = torch.mean(selected)
        std = torch.std(selected, correction=0)
        scale = std + epsilon
        result['dt_token_advantages'][index] = torch.where(active, (raw[index]-mean)/scale, 0.)
        result['dt_q_estimates'][index] = torch.where(active, q0[index]/scale, 0.)
        result['dt_v_estimates'][index] = torch.where(active, (v0[index]+mean)/scale, 0.)
        statistics.append(dict(uid=str(uid), policy_tokens=selected.numel(),
            reward_mean=float(mean), reward_std=float(std), denominator=float(scale)))
    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise RuntimeError('v12 group-normalized credit is non-finite')
        data.batch[name] = value.float()
    data.batch['advantages'] = data.batch['dt_token_advantages']
    data.batch['returns'] = data.batch['dt_q_estimates']
    data.meta_info['dt_v12_group_statistics'] = statistics
    values = data.batch['advantages'][mask]
    print(f'[DT v12 attribution then group] groups={len(statistics)} '
          f'policy_tokens={values.numel()} '
          f'A_min={float(values.min()) if values.numel() else 0.} '
          f'A_max={float(values.max()) if values.numel() else 0.} '
          f'group_std_min={min((x["reward_std"] for x in statistics), default=0.)} '
          f'epsilon={epsilon} second_normalization=False', flush=True)
    return data
