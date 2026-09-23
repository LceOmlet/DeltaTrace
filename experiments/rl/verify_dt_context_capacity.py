"""Exercise an exact 32768-token DT boundary and the original actor update.

Uses an explicitly padded capacity fixture from recorded official task rewards.
This is neither a generated task trajectory nor a task success-rate evaluation.
No model, finite rule, advantage formula, or PPO implementation is replaced.
"""
import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def capacity_fixture(original, tokenizer, alphabet, response_tokens=1024):
    """One shared factual input for DT, native backward and PPO capacity tests."""
    width = len(original['responses'])
    count = sum(original['attention_mask'][-width:])
    prompt = torch.tensor(original['input_ids'][:-width])[torch.tensor(original['attention_mask'][:-width]).bool()]
    actions = torch.tensor(original['responses'][:count])
    filler_id = tokenizer.encode(' context', add_special_tokens=False)[0]
    if response_tokens:
        assert response_tokens >= count
        actions = torch.cat((torch.full((response_tokens-count,), filler_id), actions))
    step = int(original['env_step'])
    query = alphabet.query_ids(tokenizer, current_step=step, event_step=step, max_steps=15)
    fill = 32768-len(query)-1-prompt.numel()-actions.numel()
    assert fill > 0
    row = {**original, 'responses': actions,
           'input_ids': torch.cat((torch.full((fill,), filler_id), prompt, actions)),
           'attention_mask': torch.ones(32768-len(query)-1, dtype=torch.long)}
    label = alphabet.label_ids(tokenizer)[alphabet.observed_index(float(original['rewards']))]
    factual = torch.cat((row['input_ids'], torch.tensor(query), torch.tensor([label])))
    assert factual.shape == (32768,)
    detail = dict(capacity_response_tokens=actions.numel(), synthetic_response_tokens=actions.numel()-count,
                  synthetic_filler_tokens=fill, original_response_tokens=count,
                  actor_input_tokens=row['input_ids'].numel(), query_tokens=len(query),
                  dt_input_tokens=factual.numel(), official_fixture_reward=float(original['rewards']),
                  reward_scope='Recorded official reward used as a numerical capacity coefficient; not a reward claim for the synthetic trajectory',
                  factual_input_ids_sha256=hashlib.sha256(factual.numpy().tobytes()).hexdigest(),
                  actor_input_ids_sha256=hashlib.sha256(row['input_ids'].numpy().tobytes()).hexdigest())
    return row, factual, detail, filler_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--backend', choices=('hf', 'vllm'), default='hf')
    parser.add_argument('--reshard-after-forward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--response-tokens', type=int, default=0,
                        help='Explicit synthetic action-span width; 0 keeps the recorded script actions')
    parser.add_argument('--actor-microbatch', type=int, choices=(1, 4), default=1)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--dt-repeats', type=int, default=1,
                        help='Repeat identical B4 DT calls before any update to measure warm execution')
    parser.add_argument('--parameter-offload-policy', action='store_true',
                        help='Use the installed VERL FSDP2 CPUOffloadPolicy for layer parameters')
    parser.add_argument('--tail-batch-probe', action='store_true',
                        help='After B4 warmup, measure B2/B2/B4 compiler and phase costs; skip already-tested PPO updates.')
    args = parser.parse_args()
    if args.tail_batch_probe and args.backend != 'vllm':
        parser.error('--tail-batch-probe requires the real vllm worker lifecycle')
    result = dict(scope=__doc__, context_cap=32768, minibatch=4,
                  backend=args.backend,
                  reshard_after_forward=args.reshard_after_forward,
                  actor_microbatch=args.actor_microbatch, activation_offload=args.activation_offload,
                  parameter_offload_policy=args.parameter_offload_policy, stages=[])
    if args.tail_batch_probe:
        result['scope'] = ('Tail-batch compile/phase diagnosis at exact 32768 with the original actor and '
                           'sleeping vLLM. Does not rerun or claim PPO-update verification.')
    artifacts = {}

    def stage(name):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result['stages'].append(dict(name=name, seconds=time.perf_counter()-started,
                                    host_rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                                    device_used_bytes=total-free,
                                    device_total_bytes=total,
                                    allocated=torch.cuda.memory_allocated(),
                                    peak_allocated=torch.cuda.max_memory_allocated(),
                                    peak_reserved=torch.cuda.max_memory_reserved()))
        print('STAGE', result['stages'][-1], flush=True)
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')

    def trainable_state(module):
        # Public DTensor conversion; this single-rank test owns no sharding logic.
        return {n: (p.full_tensor() if hasattr(p, 'full_tensor') else p).detach().cpu().clone()
                for n, p in module.named_parameters() if p.requires_grad}

    started = time.perf_counter()
    try:
        torch.manual_seed(2026)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.model.enable_activation_offload = args.activation_offload
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = args.actor_microbatch
        c.actor.use_dynamic_bsz = False
        c.actor.ppo_max_token_len_per_gpu = 32768
        c.actor.use_torch_compile = False
        c.actor.entropy_coeff = 0.0
        c.actor.clip_ratio_c = float('inf')
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.optimizer_offload = True
        c.actor.fsdp_config.reshard_after_forward = args.reshard_after_forward
        c.actor.fsdp_config.offload_policy = args.parameter_offload_policy
        c.rollout.name = args.backend
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.log_prob_micro_batch_size_per_gpu = args.actor_microbatch
        c.rollout.micro_batch_size = 4
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
            # Exercise the real sync and sleep before entering the DT phase.
            with worker.rollout_sharding_manager:
                pass
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
            if worker._is_offload_param:
                load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        stage('owner_init')
        from deltatrace_rollout import DeltaTraceRolloutProducer
        producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
            eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
        if args.backend == 'vllm':
            worker._deltatrace_producer = producer
        stage('producer_init')
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        original = fixture['tasks']['Sokoban']['rows'][-1]
        row, _, fixture_detail, filler_id = capacity_fixture(
            original, worker.tokenizer, producer.readout.alphabet, args.response_tokens)
        result.update(fixture_detail)
        assert result['dt_input_tokens'] == 32768
        # Exercise the actual readout's guard. It must fail before the runner.
        oversized = copy.copy(row)
        oversized['input_ids'] = torch.cat((torch.tensor([filler_id]), row['input_ids']))
        oversized['attention_mask'] = torch.ones_like(oversized['input_ids'])
        try:
            producer.attribute_episode([oversized], float(original['rewards']))
        except ValueError as exc:
            assert '32769 exceeds cap 32768; no silent truncation' in str(exc)
            result['oversize_rejected'] = str(exc)
        else:
            raise AssertionError('32769-token readout was not rejected')
        stage('oversize_rejected')
        # Retain the owner ledger to locate allocation/compute regressions.
        native_attribute = producer.runner.attribute
        ledgers = []
        def recorded_attribute(*a, **kw):
            signed, detail = native_attribute(*a, **kw)
            ledgers.append(detail)
            return signed, detail
        producer.runner.attribute = recorded_attribute
        values = []
        if args.backend == 'vllm':
            for repeat in range(args.dt_repeats):
                episodes = worker.compute_dt_token_advantages(
                    [[row] for _ in range(4)], [float(original['rewards'])]*4,
                    eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id,
                )
                values = [episode[0] for episode in episodes]
                assert producer.readout.last_report['max_readout_length'] == 32768
                result.setdefault('readouts', []).append(producer.readout.last_report)
                stage('dt_worker_batch_4' if repeat==0 else 'dt_worker_batch_4_repeat_'+str(repeat))
        if args.tail_batch_probe:
            # Diagnose the observed slow first B2 tail using the same actor,
            # 32768-token fixture, resident sleeping vLLM and owner DT runner.
            # Read the compiler's own counters; no cache reset or model patch.
            from torch._dynamo.utils import counters, compile_times
            artifacts['before'] = trainable_state(worker.actor_module_fsdp)
            artifacts['dt_batch4_warm'] = [{k: v[k].detach().cpu() for k in
                ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates')} for v in values]
            result['tail_batch_runs'] = []
            for size in (2, 2, 4):
                before = {group: dict(values) for group, values in counters.items()}
                first_ledger = len(ledgers)
                tick = time.perf_counter()
                probe = worker.compute_dt_token_advantages(
                    [[row] for _ in range(size)], [float(original['rewards'])]*size,
                    eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id,
                )
                torch.cuda.synchronize()
                assert len(probe) == size
                assert all(torch.isfinite(ep[0]['dt_token_advantages']).all() for ep in probe)
                artifacts[f'dt_batch{size}_{len(result["tail_batch_runs"])+1}'] = [
                    {k: ep[0][k].detach().cpu() for k in
                     ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates')} for ep in probe]
                after = {group: dict(values) for group, values in counters.items()}
                delta = {group: {key: value-before.get(group, {}).get(key, 0)
                                for key, value in values.items()
                                if value != before.get(group, {}).get(key, 0)}
                         for group, values in after.items()}
                result['tail_batch_runs'].append(dict(batch=size, seconds=time.perf_counter()-tick,
                    compiler_counter_delta={key: value for key, value in delta.items() if value},
                    owner_ledgers=ledgers[first_ledger:]))
                stage('tail_batch_'+str(size)+'_'+str(len(result['tail_batch_runs'])))
            result['compiler_times'] = compile_times()
            result['owner_ledgers'] = ledgers
            result['status'] = 'completed_tail_batch_diagnostic'
            return
        for index in range(4) if args.backend == 'hf' else []:
            values.append(producer.attribute_episode([row], float(original['rewards']))[0])
            assert producer.readout.last_report['max_readout_length'] == 32768
            result.setdefault('readouts', []).append(producer.readout.last_report)
            stage('dt_'+str(index))
        result['owner_ledgers'] = ledgers
        batch_data = {name: torch.stack([row[name]]*4).cuda()
                      for name in ('input_ids', 'attention_mask', 'responses')}
        batch_data['position_ids'] = batch_data['attention_mask'].cumsum(-1)-1
        for name in ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates'):
            batch_data[name] = torch.stack([v[name] for v in values]).cuda()
        batch = DataProto(batch=TensorDict(batch_data, batch_size=[4]),
                          meta_info={'temperature': 1.0, 'global_token_num': [row['input_ids'].numel()]*4})
        batch = compute_advantage(batch, adv_estimator='deltatrace', gamma=1.0)
        torch.testing.assert_close(batch.batch['advantages'], batch_data['dt_token_advantages'], rtol=0, atol=0)
        assert bool(batch.batch['advantages'].count_nonzero())
        artifacts['advantages'] = batch.batch['advantages'].cpu()
        artifacts['q'] = batch.batch['dt_q_estimates'].cpu()
        artifacts['v'] = batch.batch['dt_v_estimates'].cpu()
        artifacts['before'] = trainable_state(worker.actor_module_fsdp)
        out = worker.compute_log_prob(batch)
        batch.batch['old_log_probs'] = out.batch['old_log_probs'].cuda()
        stage('old_log_probs')
        for index in range(2):
            output = worker.update_actor(batch)
            result.setdefault('updates', []).append(output.meta_info['metrics'])
            norms = output.meta_info['metrics']['actor/grad_norm']
            assert all(torch.isfinite(torch.tensor(n)) and n > 0 for n in norms)
            stage('actor_update_'+str(index))
        artifacts['after'] = trainable_state(worker.actor_module_fsdp)
        result['changed_elements'] = sum(int((p != artifacts['before'][n]).sum()) for n, p in artifacts['after'].items())
        assert result['changed_elements'] > 0
        if args.backend == 'vllm':
            # A second native handoff must bind the updated, now nonzero LoRA.
            with worker.rollout_sharding_manager:
                manager = worker.rollout_sharding_manager.model_runner.lora_manager._adapter_manager
                adapters = manager.list_adapters()
                adapter = next(iter(adapters.values()))
                unused = sorted(set(adapter.loras) - set(manager.modules))
                nonzero_b = 0
                for layer in adapter.loras.values():
                    tensors = layer.lora_b if isinstance(layer.lora_b, list) else [layer.lora_b]
                    nonzero_b += sum(int(t.count_nonzero()) for t in tensors if t is not None)
                result['updated_native_lora'] = dict(loaded_layers=len(adapter.loras),
                    unused_layers=unused, nonzero_b_elements=nonzero_b)
                assert adapter.loras and not unused and nonzero_b > 0, result['updated_native_lora']
            stage('updated_lora_native_sync_and_sleep')
        result['ppo_source_sha256'] = hashlib.sha256((Path(os.environ['VERL_ROOT'])/'verl/trainer/ppo/core_algos.py').read_bytes()).hexdigest()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
        torch.save(artifacts, args.artifacts)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] not in ('passed', 'completed_tail_batch_diagnostic'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
