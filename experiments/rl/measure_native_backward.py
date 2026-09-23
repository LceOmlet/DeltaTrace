"""Time the pinned Qwen/VERL actor's native backward at the DT capacity shape.

Uses the same factual 32768-token event input and B4 as the capacity test.
Native checkpoint recomputation inside backward is included and reported.
Optional original PPO updates are reported separately from this cost reference.
This is not a numerical equivalence test or a task trajectory.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from reward_readout import RewardAlphabet
from verify_dt_context_capacity import capacity_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--backend', choices=('hf', 'vllm'), default='hf')
    parser.add_argument('--ppo-artifacts', type=Path,
                        help='Use recorded exact-capacity DT advantages for two original PPO updates at microbatch 4')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    result = dict(scope=__doc__, batch=4, tokens=32768, response_tokens=1024,
                  native_gradient_checkpointing=True,
                  native_activation_offload=args.activation_offload, backend=args.backend, runs=[])
    try:
        torch.manual_seed(2026)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.model.enable_gradient_checkpointing = True
        c.model.enable_activation_offload = args.activation_offload
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 4
        c.actor.ppo_max_token_len_per_gpu = 32768
        c.actor.use_torch_compile = False
        c.actor.use_dynamic_bsz = False
        c.actor.entropy_coeff = 0.0
        c.actor.clip_ratio_c = float('inf')
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.reshard_after_forward = True
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.optimizer_offload = True
        c.rollout.name = args.backend
        c.rollout.n = 1
        c.rollout.log_prob_micro_batch_size_per_gpu = 4
        c.rollout.tensor_model_parallel_size = 1
        if args.backend == 'vllm':
            c.actor.fsdp_config.param_offload = True
            c.rollout.load_format = 'safetensors'
            c.rollout.max_model_len = 32768
            c.rollout.max_num_seqs = 4
            c.rollout.max_num_batched_tokens = 32768
            c.rollout.gpu_memory_utilization = 0.75
            c.rollout.engine_kwargs.vllm.limit_mm_per_prompt = {'image': 0, 'video': 0}
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        if args.backend == 'vllm':
            with worker.rollout_sharding_manager:
                pass
            free, total = torch.cuda.mem_get_info()
            result['after_native_vllm_sleep_device_used_bytes'] = total-free
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
            load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        actor = worker.actor_module_fsdp
        actor.train()
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        alphabet = RewardAlphabet.for_task('Sokoban')
        row, factual, fixture_detail, _ = capacity_fixture(row, worker.tokenizer, alphabet, 1024)
        result.update(fixture_detail)
        actions = row['responses']
        labels = torch.tensor(alphabet.label_ids(worker.tokenizer), device='cuda')
        category = alphabet.observed_index(float(row['rewards']))
        ids = factual.repeat(4, 1).cuda()
        assert ids.shape == (4, 32768)
        mask = torch.ones_like(ids)
        predictor = torch.tensor([32766], device='cuda')
        result['phase'] = 'owner_ready'
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        for repeat in range(args.repeats):
            actor.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = actor(input_ids=ids, attention_mask=mask, use_cache=False,
                           logits_to_keep=predictor)
            assert output.logits.shape[:2] == (4, 1)
            logp = output.logits[:, 0].index_select(-1, labels).float().log_softmax(-1)
            loss = -logp[:, category].sum()
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter()-start
            result['phase'] = f'backward_{repeat}'
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print('NATIVE_BACKWARD_START', repeat, forward_seconds, flush=True)
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter()-start
            gradients = [p.grad for p in actor.parameters() if p.requires_grad and p.grad is not None]
            assert gradients
            # Read local single-rank tensors through the public DTensor API.
            finite = all(bool(torch.isfinite(g.to_local() if hasattr(g, 'to_local') else g).all())
                         for g in gradients)
            assert finite
            free, total = torch.cuda.mem_get_info()
            record = dict(repeat=repeat, forward_seconds=forward_seconds,
                          backward_seconds=backward_seconds,
                          peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          device_used_bytes=total-free, device_total_bytes=total,
                          finite_gradients=finite, loss=float(loss.detach()))
            result['runs'].append(record)
            print('NATIVE_BACKWARD_RESULT', record, flush=True)
            del output, logp, loss, gradients
        actor.zero_grad(set_to_none=True)
        if args.ppo_artifacts:
            from tensordict import TensorDict
            from verl import DataProto
            from verl.trainer.ppo.ray_trainer import compute_advantage
            artifacts = torch.load(args.ppo_artifacts, map_location='cpu', weights_only=True)
            # Remove only the separate DT query/label. This reconstructs the
            # exact 32597-token actor capacity row that produced the artifacts.
            actor_ids = row['input_ids'].repeat(4, 1).cuda()
            batch_data = dict(input_ids=actor_ids, attention_mask=torch.ones_like(actor_ids),
                              responses=actions.repeat(4, 1).cuda(),
                              position_ids=torch.arange(actor_ids.shape[1], device='cuda').repeat(4, 1),
                              dt_token_advantages=artifacts['advantages'].cuda(),
                              dt_q_estimates=artifacts['q'].cuda(), dt_v_estimates=artifacts['v'].cuda())
            batch = DataProto(batch=TensorDict(batch_data, batch_size=[4]),
                              meta_info={'temperature': 1.0, 'global_token_num': [actor_ids.shape[1]]*4})
            batch = compute_advantage(batch, adv_estimator='deltatrace', gamma=1.0)
            result['phase'] = 'ppo_old_log_probs'
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            out = worker.compute_log_prob(batch)
            batch.batch['old_log_probs'] = out.batch['old_log_probs'].cuda()
            actual_batches = []
            def record_batch(_module, _args, kwargs):
                actual_batches.append(list(kwargs['input_ids'].shape))
            hook = actor.register_forward_pre_hook(record_batch, with_kwargs=True)
            try:
                for index in range(2):
                    result['phase'] = 'ppo_update_'+str(index)
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
                    start = time.perf_counter()
                    output = worker.update_actor(batch)
                    metrics = output.meta_info['metrics']
                    assert all(torch.isfinite(torch.tensor(n)) and n > 0 for n in metrics['actor/grad_norm'])
                    result.setdefault('ppo_updates', []).append(dict(seconds=time.perf_counter()-start,
                                                                     metrics=metrics))
            finally:
                hook.remove()
            result['ppo_actual_forward_batches'] = actual_batches
            assert actual_batches == [[4, actor_ids.shape[1]]]*2, actual_batches
            if args.backend == 'vllm':
                with worker.rollout_sharding_manager:
                    pass
                result['updated_native_lora_sync_and_sleep'] = True
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
