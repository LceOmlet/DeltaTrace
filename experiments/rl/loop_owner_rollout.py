"""LOOP-owned episodes and exact policy artifacts transported into VERL."""
from pathlib import Path
from queue import Empty
import os
import site
import time
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto


class LoopOwner:
    def __init__(self, configuration, tokenizer, evaluation):
        if os.environ.get('LOOP_EXTRAS'):
            site.addsitedir(os.environ['LOOP_EXTRAS'])
        from loop_owner_recipe import compose, environment_configuration
        from loop_environment_entry import training_readout_reserve
        self.config, self.tokenizer, self.evaluation = configuration, tokenizer, evaluation
        root = Path(configuration.env.loop.owner_root)
        cfg = compose(root, [])
        self.native = OmegaConf.merge(cfg, cfg.rl.eval.overrides) if evaluation else cfg
        self.reserve = 0 if evaluation else training_readout_reserve(environment_configuration(root), tokenizer)
        self.processes = None

    def close(self):
        if self.processes is not None:
            self.processes.close()
            self.processes = None

    def collect_native_trajectories(self, collector, gen_batch, actor_rollout_wg, is_train):
        from loop_owner_worker import OwnerProcesses
        from owner_environment_transport import policy_reply
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        if self.processes is None:
            world = 1 if self.evaluation else self.config.trainer.n_gpus_per_node*self.config.trainer.nnodes
            self.processes = OwnerProcesses(OmegaConf.to_container(self.native, resolve=True),
                self.config.actor_rollout_ref.model.path, self.reserve, self.evaluation, world)
        pool = self.processes
        pool.start()
        results, records, pending = {}, {}, []
        maximum = self.config.actor_rollout_ref.rollout.max_num_seqs
        calls = tokens = 0
        started = time.monotonic()
        while len(results) < len(pool.processes):
            event = pool.event() if not pending else None
            while True:
                if event is not None:
                    if event[0] == 'result':
                        results[event[1]] = event[2]
                    elif event[0] == 'completion':
                        pending.append(event)
                    else:
                        raise RuntimeError(f'Unexpected LOOP transport event: {event[0]}')
                if len(pending) >= maximum:
                    break
                try:
                    event = pool.event(block=False)
                except Empty:
                    break
            # The owner cancellation event is authoritative. Do not submit an
            # already-cancelled HTTP-equivalent request to the GPU engine.
            pending = [e for e in pending if not pool.cancellations[e[1]].is_set()]
            requests, pending = pending[:maximum], pending[maximum:]
            if not requests:
                continue
            prompts, options = [], []
            for _, rank, key, raw in requests:
                request = dict(raw)
                prompts.append(list(request.pop('prompt')))
                request.pop('model')
                request.pop('stream')
                request['detokenize'] = True
                options.append(request)
            carrier = DataProto.from_dict(tensors=dict(input_ids=torch.zeros(len(requests), 1, dtype=torch.long)),
                non_tensors=dict(data_source=np.array(['appworld']*len(requests), dtype=object)),
                meta_info=gen_batch.meta_info)
            batch = collector.preprocess_batch(carrier, dict(raw_prompt_ids=prompts, sampling_kwargs=options))
            batch.meta_info.update(eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
            batch, padding = pad_dataproto_to_divisor(batch, actor_rollout_wg.world_size)
            output = unpad_dataproto(actor_rollout_wg.generate_sequences(batch), padding)
            for index, (_, rank, key, _) in enumerate(requests):
                reply = policy_reply(output, index)
                records[key] = output.select_idxs([index])
                pool.replies[rank].put((key, reply))
                tokens += len(reply.token_ids)
            calls += 1
            print(f'[loop_transport] calls={calls} requests={len(records)} generated_tokens={tokens} '
                  f'elapsed={time.monotonic()-started:.1f}s completed_ranks={len(results)}', flush=True)
        self.records = records
        rows = []
        for rank in sorted(results):
            scenarios, groups = results[rank]
            for scenario, group in zip(scenarios, groups):
                uid = str(uuid.uuid4())
                rows.extend((scenario, rollout, uid) for rollout in group)
        return self.to_batch(rows, records)

    def to_batch(self, rows, records):
        from agent_system.reward_manager.episode import EpisodeRewardManager
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import pad_2d_list_to_length
        cap = self.native.rl.learning_max_seq_len or self.tokenizer.model_max_length
        # The source artifact and truncation boundary are LOOP's PolicyTokenInfo
        # and learning_max_seq_len (_get_tensors in the author's trainer).
        # No message reconstruction or tokenization takes place here.
        artifacts = [r.policy_token_info for _, r, _ in rows]
        lengths = [min(len(a.tokens), cap) for a in artifacts]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        ids = pad_2d_list_to_length([a.tokens[:cap] for a in artifacts], pad, max_length=width)
        mask = pad_2d_list_to_length([a.is_output[:cap] for a in artifacts], 0, max_length=width)
        attention = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
        logprobs = pad_2d_list_to_length([a.log_probs[:cap] for a in artifacts], 0., max_length=width)
        logprobs = torch.where(mask.bool(), logprobs, 0.)
        # A one-token prompt leaves the exact full trajectory intact; loss_mask
        # is the native per-token action mask, including later observations.
        assert not mask[:, 0].any()
        rewards = torch.zeros((len(rows), width-1))
        for index, (_, rollout, _) in enumerate(rows):
            rewards[index, lengths[index]-2] = rollout.ret
        metadata = dict(data_source=np.array(['appworld']*len(rows), dtype=object),
            uid=np.array([uid for _, _, uid in rows], dtype=object),
            traj_uid=np.array([str(uuid.uuid4()) for _ in rows], dtype=object),
            episode_rewards=np.array([r.ret for _, r, _ in rows]),
            episode_lengths=np.array([r.appworld_rollout_data.eval_result.num_interactions for _, r, _ in rows]),
            tool_callings=np.array([r.tool_execution_count for _, r, _ in rows]))
        output = DataProto.from_dict(tensors=dict(input_ids=ids, prompts=ids[:, :1], responses=ids[:, 1:],
            attention_mask=attention.long(), position_ids=compute_position_id_with_mask(attention),
            loss_mask=mask.long(), response_mask=mask[:, 1:].long(),
            rm_scores=rewards, rollout_log_probs=logprobs[:, 1:]), non_tensors=metadata)
        output.meta_info['multi_turn'] = True
        if self.evaluation:
            from hydra.utils import instantiate
            summary = instantiate(self.native.rl.eval.summaries)
            metrics = summary.summary([s for s, _, _ in rows], [r for _, r, _ in rows])
            print(f'[loop_evaluation] {metrics}', flush=True)
            return output
        sources, maps = [], []
        for index, (_, rollout, _) in enumerate(rows):
            mapping = []
            turns = rollout.completion_requests
            count = rollout.appworld_rollout_data.eval_result.num_tests
            for step, (key, start, length) in enumerate(turns):
                row = records[key]
                row.non_tensor_batch.update(traj_uid=np.array([metadata['traj_uid'][index]], dtype=object),
                    env_step=np.array([step]), active_masks=np.array([True]),
                    episode_rewards=np.array([rollout.ret]), episode_lengths=np.array([len(turns)]),
                    rewards=np.array([rollout.ret if step == len(turns)-1 else 0.]),
                    appworld_num_tests=np.array([count]))
                row.batch['token_level_rewards'] = EpisodeRewardManager(self.tokenizer, 0)(row)
                length = max(0, min(length, lengths[index]-start))
                if length:
                    assert ids[index, start:start+length].tolist() == row.batch['responses'][0, :length].tolist()
                    assert mask[index, start:start+length].all()
                    mapping.append((len(sources), start-1, length))
                sources.append(row)
            maps.append(mapping)
        self.credit_responses = DataProto.concat(sources) if sources else None
        array = np.empty(len(maps), dtype=object)
        array[:] = maps
        output.non_tensor_batch['dt_response_slices'] = array
        print(f'[loop_trajectory] trajectories={len(rows)} responses={len(sources)} '
              f'policy_tokens={int(mask.sum())} max_context={max(lengths)}', flush=True)
        return output
