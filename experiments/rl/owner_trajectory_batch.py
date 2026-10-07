"""Translate native trajectory artifacts into VERL's existing multi-turn batch.

SkyRL owns sequence assembly, padding, reward positions and observation masks.
VERL owns DataProto, positions, batching, the actor and its optimizer. DT still
uses the collected factual response rows; only their destination indices change.
"""
import numpy as np
import torch


def native_credit_response_batch(tokenizer, rows):
    """Pack an owner's exact training-token artifacts for the existing DT API."""
    from verl import DataProto
    from verl.utils.torch_functional import postprocess_data, pad_2d_list_to_length
    from verl.utils.model import compute_position_id_with_mask
    from agent_system.reward_manager.episode import EpisodeRewardManager
    if not rows:
        return None
    width = max(len(row['prompt_ids']) for row in rows)
    prompts, masks = zip(*(postprocess_data(torch.tensor([row['prompt_ids']]),
        torch.ones(1, len(row['prompt_ids']), dtype=torch.long), width,
        tokenizer.pad_token_id, left_pad=True, truncation='error') for row in rows))
    prompts, prompt_mask = torch.cat(prompts), torch.cat(masks)
    responses = pad_2d_list_to_length([row['response_ids'] for row in rows], tokenizer.pad_token_id)
    response_mask = torch.arange(responses.shape[1])[None, :] < torch.tensor(
        [len(row['response_ids']) for row in rows])[:, None]
    attention = torch.cat((prompt_mask, response_mask.long()), -1)
    metadata = {key: np.array([row[key] for row in rows], dtype=object if key in ('traj_uid', 'data_source') else None)
                for key in rows[0] if key not in ('prompt_ids', 'response_ids')}
    result = DataProto.from_dict(tensors=dict(input_ids=torch.cat((prompts, responses), -1),
        prompts=prompts, responses=responses, attention_mask=attention,
        position_ids=compute_position_id_with_mask(attention)), non_tensors=metadata)
    result.batch['token_level_rewards'] = EpisodeRewardManager(tokenizer, 0)(result)
    return result


def sql_trajectory_batch(manager, collector, **collected):
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    from agent_system.reward_manager.episode import EpisodeRewardManager
    from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

    trajectories = [event.trajectory for event in manager.current]
    ids, attention, response_mask, rewards, loss_mask, logprobs, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            collector.tokenizer.pad_token_id,
            [t.prompt_ids for t in trajectories], [t.response_ids for t in trajectories],
            [t.reward for t in trajectories], [t.loss_mask for t in trajectories],
            [t.rollout_logprobs for t in trajectories]))
    response_width = response_mask.shape[-1]
    prompt_width = ids.shape[-1] - response_width

    # The original gather function assigns episode/group identity and metrics.
    # Keep its response table only on the driver, never in RPC metadata.
    response_data = collector.gather_rollout_data(**collected)
    source_by_identity = {(str(uid), int(step)): i for i, (uid, step) in enumerate(zip(
        response_data.non_tensor_batch['traj_uid'], response_data.non_tensor_batch['env_step']))}
    dt = collector.config.algorithm.adv_estimator == 'deltatrace'
    if dt:
        response_data.batch['token_level_rewards'] = EpisodeRewardManager(
            collector.tokenizer, num_examine=0)(response_data)
        manager.credit_responses = response_data
    first_indices, maps = [], []
    for index, (trajectory, turns) in enumerate(zip(trajectories, manager.rollouts.records)):
        uid = str(collected['traj_uid'][index])
        first_indices.append(source_by_identity[(uid, 0)])
        mapping = []
        for step, turn in enumerate(turns):
            source_index = source_by_identity[(uid, step)]
            start = turn['start'] - len(trajectory.prompt_ids)
            length = turn['end'] - turn['start']
            assert trajectory.response_ids[start:start+length] == turn['output_ids'][:length]
            # The owner may restore a sampled EOS at the final boundary. A
            # synthetic EOS has no generated source token and receives no DT
            # attribution. Its native mask/reward are not rewritten here.
            if step == len(turns)-1:
                remaining = trajectory.response_ids[start:]
                if remaining == turn['output_ids']:
                    length = len(remaining)
            destination = response_width - len(trajectory.response_ids) + start
            mapping.append((source_index, destination, length))
        maps.append(mapping)
    metadata = response_data.select_idxs(first_indices).non_tensor_batch.copy()
    for key in list(metadata):
        if key.startswith('owner_'):
            metadata.pop(key)  # Per-request transport fields do not describe a full trajectory.
    if dt:
        mapping_array = np.empty(len(maps), dtype=object)
        mapping_array[:] = maps
        metadata['dt_response_slices'] = mapping_array
    tensors = dict(input_ids=ids, prompts=ids[:, :prompt_width], responses=ids[:, prompt_width:],
        attention_mask=attention, position_ids=compute_position_id_with_mask(attention),
        loss_mask=torch.cat((torch.zeros_like(ids[:, :prompt_width]), loss_mask), dim=-1),
        response_mask=response_mask, rm_scores=rewards, rollout_log_probs=logprobs)
    output = DataProto.from_dict(tensors=tensors, non_tensors=metadata)
    output.meta_info['multi_turn'] = True
    print(f'[owner_trajectory] trajectories={len(output)} responses={len(response_data)} '
          f'policy_tokens={int(loss_mask.sum())} context_tokens={int(attention.sum())}', flush=True)
    return output


def trajectory_credit(data, response_data, worker_group, *, eos_token_id, pad_token_id):
    from verl import DataProto
    from dt_training_batch import CREDIT_KEYS, compute_training_credit
    import os

    if 'dt_direct_target_artifact' in data.non_tensor_batch:
        from dt_training_batch import compute_direct_target_credit
        return compute_direct_target_credit(data, worker_group,
            eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    if os.environ.get('DT_TARGET_SEMANTICS') == 'native_joint_action_target':
        raise RuntimeError('The native rollout did not retain its real action target artifact')

    result = {key: torch.zeros_like(data.batch['responses'], dtype=torch.float32) for key in CREDIT_KEYS}
    if response_data is None:
        # Native reset failures can return no actions at all.
        assert not any(data.non_tensor_batch['dt_response_slices'])
        return DataProto.from_dict(tensors=result)
    # Native trajectory truncation can remove a later reward-bearing action
    # from the training tensor. Its real reward remains in response_data; only
    # actions with retained native token slices require a DT request.
    indices = torch.tensor(sorted({source
        for slices in data.non_tensor_batch['dt_response_slices']
        for source, _, length in slices if length > 0}), dtype=torch.long)
    values = compute_training_credit(response_data, worker_group,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id, source_indices=indices)
    for row, slices in enumerate(data.non_tensor_batch['dt_response_slices']):
        for source, start, length in slices:
            for key in CREDIT_KEYS:
                result[key][row, start:start+length] = values.batch[key][source, :length]
    return DataProto.from_dict(tensors=result)
