"""Real Qwen/DT/native PPO on saved task tokens; not fresh task evaluation."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from agent_system.reward_manager.episode import EpisodeRewardManager
from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty, compute_advantage
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verify_author_rollout import configuration
from reward_readout import EventRatioReadout

root = Path(os.environ['DT_RUNTIME_ROOT'])
audit = root / 'receipts/upstream-alignment-20260929'
fixtures = json.loads((root / 'receipts/rollout-fixtures.json').read_text())
result = dict(scope=__doc__, dt_batch=4, actor_microbatch=4, actor_minibatch=64,
              context_cap=32768, tasks={}, phase='initialization')
output = audit / 'native-minibatch64-real-model.json'
started = time.perf_counter()

def record(phase):
    result.update(phase=phase, seconds=time.perf_counter() - started)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(phase, flush=True)

try:
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    cfg = configuration().actor_rollout_ref
    cfg.actor.ppo_mini_batch_size = 64
    cfg.rollout.name = 'hf'  # No rollout in a replay of saved token artifacts.
    worker = ActorRolloutRefWorker(cfg, 'actor_rollout')
    worker.init_model()
    record('model_ready')
    for task in ('Sokoban', 'Webshop', 'AppWorld'):
        row = [r for r in fixtures['tasks'][task]['rows'] if r['active_masks']][-1]
        tensors = {name: torch.tensor([row[name]] * 4, dtype=torch.long)
                   for name in ('input_ids', 'attention_mask', 'responses')}
        width = tensors['responses'].shape[1]
        tensors['prompts'] = tensors['input_ids'][:, :-width]
        tensors['position_ids'] = compute_position_id_with_mask(tensors['attention_mask'])
        tensors['response_mask'] = tensors['attention_mask'][:, -width:]
        data = DataProto.from_dict(tensors=tensors, non_tensors={
            'rewards': np.array([row['rewards']] * 4),
            'episode_rewards': np.array([row['rewards']] * 4),
            'episode_lengths': np.array([1] * 4),
            'traj_uid': np.array([f'{task}-{i}' for i in range(4)], dtype=object),
            'env_step': np.array([row['env_step']] * 4),
            'active_masks': np.ones(4, dtype=bool),
            # Explicit interface fixture: the first row exercises the native
            # invalid-action penalty; this is not a task-parser accuracy claim.
            'is_action_valid': np.array([[False], [True], [True], [True]]),
            'data_source': np.array([task] * 4, dtype=object),
        }, meta_info={'temperature': 1.0,
                     'global_token_num': tensors['attention_mask'].sum(-1).tolist()})
        data.batch['token_level_scores'] = EpisodeRewardManager(worker.tokenizer, 0)(data)
        data, _ = apply_invalid_action_penalty(data, cfg.actor.invalid_action_penalty_coef)
        data.batch['token_level_rewards'] = data.batch['token_level_scores']
        os.environ['DT_TASK'] = task
        if hasattr(worker, '_deltatrace_producer'):
            producer = worker._deltatrace_producer
            producer.readout = EventRatioReadout(
                producer.runner, worker.tokenizer, task=task, max_steps=15,
                packed_answer_targets=producer.packed_answer_targets,
                invalid_action_penalty_coef=cfg.actor.invalid_action_penalty_coef)
        record(task + ':dt_start')
        tick = time.perf_counter()
        credit = worker.compute_dt_token_advantages(
            data, eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
        dt_seconds = time.perf_counter() - tick
        data = compute_advantage(data.union(credit), 'deltatrace')
        mask = data.batch['response_mask'].bool()
        q = data.batch['dt_q_estimates']
        expected = data.batch['token_level_rewards'].sum(-1)[:, None].expand_as(q)
        torch.testing.assert_close(q[mask], expected[mask])
        assert all(torch.isfinite(credit.batch[key]).all() for key in credit.batch.keys())
        assert not data.batch['advantages'][~mask].any()
        # Repeat the four saved interface fixtures through the native DataProto
        # API to exercise one full official minibatch of 64 (16 microbatches).
        # This tests accumulation, not 64 independently sampled trajectories.
        data = data.repeat(repeat_times=16, interleave=True)
        data.meta_info['global_token_num'] = data.batch['attention_mask'].sum(-1).tolist()
        record(task + ':native_ppo_start')
        old = worker.compute_log_prob(data)
        data.batch['old_log_probs'] = old.batch['old_log_probs']
        tick = time.perf_counter()
        update = worker.update_actor(data)
        result['tasks'][task] = dict(
            dt_seconds=dt_seconds, update_seconds=time.perf_counter() - tick,
            input_shape=list(tensors['input_ids'].shape),
            actor_rows=len(data),
            q_per_row=q[:, 0].tolist(),
            nonzero_advantages=int(data.batch['advantages'].count_nonzero()),
            categories=len(worker._deltatrace_producer.readout.alphabet.values),
            metrics=update.meta_info['metrics'],
            dt_conservation_failures=worker._deltatrace_producer.readout.last_report['conservation_failures'],
        )
        record(task + ':complete')
    result['status'] = 'completed_credit_interface_and_native_update_not_task_training'
except Exception as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    record('finished')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
