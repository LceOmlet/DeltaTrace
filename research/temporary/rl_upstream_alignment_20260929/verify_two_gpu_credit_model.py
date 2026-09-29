"""Actual two-rank VERL/DT on saved task tokens; not fresh task performance."""
import json
import os
from pathlib import Path
import time

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from agent_system.reward_manager.episode import EpisodeRewardManager
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty, compute_advantage
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verify_author_rollout import configuration
from reward_readout import EventRatioReadout
from dt_training_batch import compute_training_credit


@ray.remote
class ProbeWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_rollout(self):
        # Exercise the native tensor LoRA synchronization and vLLM sleep owner.
        with self.rollout_sharding_manager:
            pass
        return self.resources()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resources(self):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(rank=self.rank, physical_used_bytes=total-free, total_bytes=total,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def lora_metadata(self, materialize=False):
        from peft.utils.save_and_load import get_peft_model_state_dict
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        import hashlib
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        try:
            params = get_peft_model_state_dict(self.actor_module_fsdp)
            info = []
            for name, param in params.items():
                item = dict(name=name, shape=list(param.shape), dtype=str(param.dtype), device=str(param.device),
                    local_shape=list(param.to_local().shape) if hasattr(param, 'to_local') else list(param.shape),
                    placements=str(getattr(param, 'placements', None)))
                if materialize:
                    print(json.dumps(dict(phase='native_lora_full_tensor', rank=self.rank, **item)), flush=True)
                    full = param.full_tensor() if hasattr(param, 'full_tensor') else param
                    item['sha256'] = hashlib.sha256(full.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                info.append(item)
            return info
        finally:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_task(self, task):
        os.environ['DT_TASK'] = task
        if hasattr(self, '_deltatrace_producer'):
            producer = self._deltatrace_producer
            producer.readout = EventRatioReadout(producer.runner, self.tokenizer,
                task=task, max_steps=15, packed_answer_targets=producer.packed_answer_targets,
                invalid_action_penalty_coef=self.config.actor.invalid_action_penalty_coef)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def credit_report(self):
        report = self._deltatrace_producer.readout.last_report
        return dict(rank=self.rank, contrasts=report['event_contrasts'],
            calls=report['finite_trace_calls'], seconds=report['seconds'],
            conservation_failures=report['conservation_failures'])


root = Path(os.environ['DT_RUNTIME_ROOT'])
audit = root / 'receipts/upstream-alignment-20260929'
fixtures = json.loads((root / 'receipts/rollout-fixtures.json').read_text())
diagnose = os.environ.get('DT_DIAG_LORA_ONLY') == '1'
capacity = os.environ.get('DT_CAPACITY_TEST') == '1'
output = audit / ('native-two-gpu-capacity32k.json' if capacity else
    'native-two-gpu-lora-diagnostic.json' if diagnose else 'native-two-gpu-credit-native-lora.json')
global_minibatch = 8 if capacity else 64
result = dict(scope=__doc__, world_size=2, dt_local_batch=4, actor_local_microbatch=4,
    actor_global_minibatch=global_minibatch, context_cap=32768, tasks={}, status='running')
if capacity:
    result['scope'] = 'Synthetic exact32768 DT capacity with native sleeping vLLM on two GPUs; B4 per GPU. Not natural task lengths or a formal optimizer minibatch64 timing.'
started = time.perf_counter()


def record(phase):
    result.update(phase=phase, seconds=time.perf_counter()-started)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(phase=phase, seconds=result['seconds'])), flush=True)


ray.init(num_cpus=8, include_dashboard=False)
try:
    cfg = configuration().actor_rollout_ref
    cfg.actor.ppo_mini_batch_size = global_minibatch
    cfg.rollout.name = 'vllm' if capacity else 'hf'
    group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
        RayClassWithInitArgs(ProbeWorker, cfg, 'actor_rollout'))
    group.init_model()
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    record('model_ready')
    if capacity:
        result['native_rollout_sync'] = group.sync_rollout()
        record('native_rollout_sync_complete')
    if diagnose:
        result['lora_metadata'] = group.lora_metadata()
        record('lora_metadata_before_collectives')
        assert [[(x['name'], x['shape']) for x in values] for values in result['lora_metadata']][0] == [
            (x['name'], x['shape']) for x in result['lora_metadata'][1]]
        result['lora_gather'] = group.lora_metadata(materialize=True)
        assert [x['sha256'] for x in result['lora_gather'][0]] == [x['sha256'] for x in result['lora_gather'][1]]
        record('lora_gather_exact_rank_match')
    for task in (() if diagnose else ('Sokoban',) if capacity else ('Sokoban', 'Webshop', 'AppWorld')):
        group.set_task(task)
        row = [r for r in fixtures['tasks'][task]['rows'] if r['active_masks']][-1]
        if capacity:
            from verify_dt_context_capacity import capacity_fixture
            from reward_readout import RewardAlphabet
            row, factual, detail, _ = capacity_fixture(row, tokenizer,
                RewardAlphabet.for_task(task, 15, cfg.actor.invalid_action_penalty_coef), response_tokens=512)
            result['capacity_fixture'] = detail
            row = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in row.items()}
            assert len(factual) == 32768
        tensors = {key: torch.tensor([row[key]]*8, dtype=torch.long)
            for key in ('input_ids', 'attention_mask', 'responses')}
        width = tensors['responses'].shape[1]
        tensors['prompts'] = tensors['input_ids'][:, :-width]
        tensors['position_ids'] = compute_position_id_with_mask(tensors['attention_mask'])
        tensors['response_mask'] = tensors['attention_mask'][:, -width:]
        data = DataProto.from_dict(tensors=tensors, non_tensors=dict(
            rewards=np.array([row['rewards']]*8), episode_rewards=np.array([row['rewards']]*8),
            episode_lengths=np.ones(8, dtype=int), traj_uid=np.array([f'{task}-{i}' for i in range(8)], dtype=object),
            env_step=np.full(8, row['env_step']), active_masks=np.ones(8, dtype=bool),
            is_action_valid=np.array([[False]]+[[True]]*7),
            data_source=np.array([task]*8, dtype=object)),
            meta_info={'temperature': 1.0, 'global_token_num': tensors['attention_mask'].sum(-1).tolist()})
        data.batch['token_level_scores'] = EpisodeRewardManager(tokenizer, 0)(data)
        data, _ = apply_invalid_action_penalty(data, cfg.actor.invalid_action_penalty_coef)
        data.batch['token_level_rewards'] = data.batch['token_level_scores']
        record(task+':dt_start')
        tick = time.perf_counter()
        credit = compute_training_credit(data, group, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id)
        dt_seconds = time.perf_counter()-tick
        data = compute_advantage(data.union(credit), 'deltatrace')
        mask = data.batch['response_mask'].bool()
        expected = data.batch['token_level_rewards'].sum(-1)[:, None].expand_as(data.batch['dt_q_estimates'])
        torch.testing.assert_close(data.batch['dt_q_estimates'][mask], expected[mask])
        assert all(torch.isfinite(credit.batch[key]).all() for key in credit.batch.keys())
        assert data.batch['advantages'].count_nonzero() and not data.batch['advantages'][~mask].any()
        result['tasks'][task] = dict(dt_seconds=dt_seconds, ranks=group.credit_report(),
            input_shape=list(tensors['input_ids'].shape), nonzero_advantages=int(data.batch['advantages'].count_nonzero()))
        if capacity:
            result['tasks'][task]['resources_after_dt'] = group.resources()
        # Native accumulation: 64 global rows, 32 per rank, B4 microbatch.
        data = data.repeat(repeat_times=global_minibatch//8, interleave=True)
        data.meta_info['global_token_num'] = data.batch['attention_mask'].sum(-1).tolist()
        record(task+':native_old_logprob')
        old = group.compute_log_prob(data)
        data.batch['old_log_probs'] = old.batch['old_log_probs']
        record(task+':native_update')
        tick = time.perf_counter()
        update = group.update_actor(data)
        metrics = update.meta_info['metrics']
        result['tasks'][task].update(update_seconds=time.perf_counter()-tick, metrics=metrics)
        if capacity:
            result['tasks'][task]['resources_after_update'] = group.resources()
        assert all(np.isfinite(x) and x > 0 for x in metrics['actor/grad_norm'])
        record(task+':complete')
    result['status'] = 'completed_saved_token_credit_and_native_updates'
except Exception as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    record('finished')
    ray.shutdown()
