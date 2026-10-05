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
        from verl.protocol import pad_dataproto_to_divisor
        from ray.util.actor_pool import ActorPool
        if self.processes is None:
            world = 1 if self.evaluation else self.config.trainer.n_gpus_per_node*self.config.trainer.nnodes
            self.processes = OwnerProcesses(OmegaConf.to_container(self.native, resolve=True),
                self.config.actor_rollout_ref.model.path, self.reserve, self.evaluation, world)
        pool = self.processes
        if hasattr(collector, 'async_rollout_manager'):
            from loop_async_transport import collect
            return collect(self, collector, gen_batch)
        pool.start()
        results, records, pending = {}, {}, []
        calls = tokens = 0
        ready = 0
        # The original Functor's name is the original wire method. VERL's
        # single-worker helper retains its fused/non-fused routing.
        wire_method = type(actor_rollout_wg.generate_sequences).__name__
        engines = ActorPool(actor_rollout_wg.workers)

        def submit(actor, item):
            # Cancellation comes from LOOP. Recheck when Ray allocates an
            # actor, because requests can wait in ActorPool's native queue.
            ranks = item.non_tensor_batch['loop_reply_rank']
            valid = item.non_tensor_batch['loop_reply_valid']
            keep = [i for i, rank in enumerate(ranks)
                    if valid[i] and not pool.cancellations[int(rank)].is_set()]
            item = item.select_idxs(torch.tensor(keep, dtype=torch.long))
            if keep:
                ref = actor_rollout_wg._execute_remote_single_worker(actor, wire_method, item)
            else:
                # VERL's generator requires a nonempty batch. Preserve the
                # empty transport artifact without invoking that generator.
                import ray
                ref = ray.put(item)
            # A native Ray future wakes the same existing queue reader. It
            # carries no result payload; ActorPool owns collection/order.
            ref.future().add_done_callback(lambda _: pool.output.put(('engine_ready',)))
            return ref

        started = time.monotonic()
        while len(results) < len(pool.processes) or engines.has_next() or pending:
            event = None if pending and engines.has_free() else pool.event()
            while True:
                if event is not None:
                    if event[0] == 'result':
                        results[event[1]] = event[2]
                    elif event[0] == 'completion':
                        pending.append(event)
                    elif event[0] == 'engine_ready':
                        ready += 1
                    else:
                        raise RuntimeError(f'Unexpected LOOP transport event: {event[0]}')
                try:
                    event = pool.queued_event()
                except Empty:
                    break
            # The owner cancellation event is authoritative. Do not submit an
            # already-cancelled HTTP-equivalent request to the GPU engine.
            pending = [e for e in pending if not pool.cancellations[e[1]].is_set()]
            # max_num_seqs belongs to each vLLM engine, after original VERL
            # DP dispatch. Submit all already queued owner requests; limiting
            # the global RPC to one engine's cap halves two-engine occupancy.
            # ActorPool owns availability. While both actors are busy,
            # retain exact requests here rather than freezing each new
            # arrival into its own queued RPC. No timer or batch cap.
            if engines.has_free():
                requests, pending = pending, []
            else:
                requests = []
            if requests:
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
                batch.non_tensor_batch.update(
                    loop_reply_rank=np.array([r for _, r, _, _ in requests]),
                    loop_reply_key=np.array([k for _, _, k, _ in requests], dtype=object))
                batch, padding = pad_dataproto_to_divisor(batch, actor_rollout_wg.world_size)
                batch.non_tensor_batch['loop_reply_valid'] = np.arange(len(batch)) < len(requests)
                for item in batch.chunk(actor_rollout_wg.world_size):
                    engines.submit(submit, item)
                calls += 1
            while ready:
                # The owner callback says a native ref is ready, so there
                # is no polling interval and no wait for the other rank.
                output = engines.get_next_unordered(timeout=0)
                ready -= 1
                for index in range(len(output)):
                    rank = int(output.non_tensor_batch['loop_reply_rank'][index])
                    key = output.non_tensor_batch['loop_reply_key'][index]
                    reply = policy_reply(output, index)
                    records[key] = output.select_idxs([index])
                    pool.replies[rank].put((key, reply))
                    tokens += len(reply.token_ids)
                print(f'[loop_transport] calls={calls} rpc_mode=actor_pool '
                      f'batch_requests={len(output)} queued_requests={pool.output.qsize()} '
                      f'requests={len(records)} generated_tokens={tokens} '
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
