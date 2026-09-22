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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--reshard-after-forward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--response-tokens', type=int, default=0,
                        help='Explicit synthetic action-span width; 0 keeps the recorded script actions')
    args = parser.parse_args()
    result = dict(scope=__doc__, context_cap=32768, minibatch=4,
                  reshard_after_forward=args.reshard_after_forward, stages=[])
    artifacts = {}

    def stage(name):
        torch.cuda.synchronize()
        result['stages'].append(dict(name=name, seconds=time.perf_counter()-started,
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
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.ppo_max_token_len_per_gpu = 32768
        c.actor.use_torch_compile = False
        c.actor.entropy_coeff = 0.0
        c.actor.clip_ratio_c = float('inf')
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.optimizer_offload = True
        c.actor.fsdp_config.reshard_after_forward = args.reshard_after_forward
        c.rollout.name = 'hf'
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.log_prob_micro_batch_size_per_gpu = 1
        c.rollout.micro_batch_size = 4
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        stage('owner_init')
        from deltatrace_rollout import DeltaTraceRolloutProducer
        producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
            eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
        stage('producer_init')
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        original = fixture['tasks']['Sokoban']['rows'][-1]
        width = len(original['responses'])
        count = sum(original['attention_mask'][-width:])
        prompt = torch.tensor(original['input_ids'][:-width])[torch.tensor(original['attention_mask'][:-width]).bool()]
        actions = torch.tensor(original['responses'][:count])
        filler_id = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
        if args.response_tokens:
            assert args.response_tokens >= count
            actions = torch.cat((torch.full((args.response_tokens-count,), filler_id), actions))
        result.update(capacity_response_tokens=actions.numel(),
                      synthetic_response_tokens=actions.numel()-count,
                      reward_scope='Recorded official reward used as a numerical capacity coefficient; not a reward claim for the synthetic trajectory')
        step = int(original['env_step'])
        query = producer.readout.alphabet.query_ids(worker.tokenizer, current_step=step, event_step=step, max_steps=15)
        fill = 32768 - len(query) - 1 - prompt.numel() - actions.numel()
        assert fill > 0
        row = {**original, 'responses': actions,
               'input_ids': torch.cat((torch.full((fill,), filler_id), prompt, actions)),
               'attention_mask': torch.ones(32768-len(query)-1, dtype=torch.long)}
        result.update(synthetic_filler_tokens=fill, original_response_tokens=count,
                      actor_input_tokens=row['input_ids'].numel(), query_tokens=len(query),
                      dt_input_tokens=row['input_ids'].numel()+len(query)+1,
                      official_fixture_reward=float(original['rewards']))
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
        for index in range(4):
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
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
