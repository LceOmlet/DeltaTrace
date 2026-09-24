"""Compare bounded interaction prefixes through the actual author collector.

Both sides use the same initialized official VERL/vLLM worker and official task
services. Greedy decoding isolates collector changes. This is a fixed-policy
regression fixture, not a trained success-rate estimate, DT/PPO update test, or
a replacement rollout implementation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from agent_system.environments import env_manager
from agent_system.multi_turn_rollout import rollout_loop


def configuration(response_cap=512):
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    settings = {
        'model.path': os.environ['MODEL_PATH'], 'model.lora_rank': 1,
        'model.lora_alpha': 2, 'model.trust_remote_code': True,
        'model.enable_activation_offload': True, 'actor.strategy': 'fsdp2',
        'actor.ppo_mini_batch_size': 4, 'actor.ppo_micro_batch_size_per_gpu': 4,
        'actor.use_dynamic_bsz': False, 'actor.ppo_max_token_len_per_gpu': 32768,
        'actor.use_torch_compile': False, 'actor.optim.total_training_steps': 3,
        'actor.fsdp_config.model_dtype': 'bfloat16',
        'actor.fsdp_config.optimizer_offload': True,
        'actor.fsdp_config.reshard_after_forward': True,
        'actor.fsdp_config.offload_policy': True, 'actor.fsdp_config.param_offload': True,
        'rollout.name': 'vllm', 'rollout.n': 1, 'rollout.tensor_model_parallel_size': 1,
        'rollout.log_prob_micro_batch_size_per_gpu': 4, 'rollout.micro_batch_size': 4,
        'rollout.prompt_length': 32768-response_cap-256, 'rollout.response_length': response_cap,
        'rollout.load_format': 'safetensors', 'rollout.max_model_len': 32768,
        'rollout.max_num_seqs': 32, 'rollout.max_num_batched_tokens': 32768,
        'rollout.gpu_memory_utilization': .75, 'rollout.enforce_eager': True,
        'rollout.engine_kwargs.vllm.limit_mm_per_prompt': {'image': 0, 'video': 0},
    }
    for key, value in settings.items():
        OmegaConf.update(cfg.actor_rollout_ref, key, value, force_add=True)
    cfg.data.train_batch_size, cfg.data.val_batch_size = 4, 1
    cfg.data.max_prompt_length, cfg.data.max_response_length = 32768-response_cap-256, response_cap
    cfg.data.truncation, cfg.data.return_raw_chat = 'error', True
    cfg.data.apply_chat_template_kwargs = {'enable_thinking': False}
    cfg.algorithm.adv_estimator = 'deltatrace'
    cfg.env.seed, cfg.env.max_steps, cfg.env.history_length = 0, 15, 2
    cfg.env.rollout.n = 1
    cfg.env.resources_per_worker.runtime_env = {'env_vars': {
        'CUDA_VISIBLE_DEVICES': '', 'MACA_VISIBLE_DEVICES': ''}}
    return cfg


def trace_rows(episodes):
    result = []
    for episode in episodes:
        rows = []
        for row in episode:
            if not row['active_masks']:
                continue
            width = row['responses'].numel()
            mask = row['attention_mask'].bool()
            prompt = row['input_ids'][:-width][mask[:-width]].tolist()
            response = row['responses'][mask[-width:]].tolist()
            rows.append(dict(prompt=prompt, response=response, reward=float(row['rewards'])))
        result.append(rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-collector', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=2,
                   help='Bound the measured prefix; environment task horizon stays 15.')
    p.add_argument('--response-cap', type=int, default=512)
    p.add_argument('--tasks', nargs='+', choices=['Sokoban', 'Webshop'], default=['Sokoban', 'Webshop'])
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('pinned_author_collector', args.author_collector)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    started = time.perf_counter()
    result = dict(scope=__doc__, status='running', context_cap=32768, episodes_per_task=4,
                  response_cap=args.response_cap, task_horizon=15, measured_rounds=args.rounds, runs=[], imports={
        'candidate_collector': rollout_loop.__file__, 'manager': env_manager.__file__,
        'author_collector': str(args.author_collector),
        'author_sha256': hashlib.sha256(args.author_collector.read_bytes()).hexdigest()})

    def record(phase, **fields):
        print(json.dumps(dict(phase=phase, elapsed=time.perf_counter()-started, **fields)), flush=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')

    try:
        torch.manual_seed(2026)
        cfg = configuration(args.response_cap)
        result['configuration'] = OmegaConf.to_container(cfg, resolve=True)
        worker = ActorRolloutRefWorker(cfg.actor_rollout_ref, 'actor_rollout')
        worker.init_model()
        record('model_ready')
        ray.init(num_cpus=8, num_gpus=0, include_dashboard=False,
                 _temp_dir=os.environ['RAY_TMPDIR'], ignore_reinit_error=False)
        engine_calls = []
        owner_generate = worker.rollout.inference_engine.generate

        def measured_generate(*a, **kw):
            tick = time.perf_counter()
            outputs = owner_generate(*a, **kw)
            torch.cuda.synchronize()
            entry = dict(seconds=time.perf_counter()-tick, requests=len(outputs),
                         prompt_tokens=sum(len(x.prompt_token_ids) for x in outputs),
                         cached_tokens=sum(x.num_cached_tokens for x in outputs),
                         generated_tokens=sum(len(y.token_ids) for x in outputs for y in x.outputs))
            engine_calls.append(entry)
            args.output.with_name(f'{args.output.stem}-request-{len(engine_calls)}.json').write_text(
                json.dumps([dict(prompt_ids=x.prompt_token_ids, outputs=[dict(
                    ids=y.token_ids, text=y.text, finish_reason=y.finish_reason, stop_reason=y.stop_reason
                ) for y in x.outputs]) for x in outputs], ensure_ascii=False)+'\n')
            record('engine_generate', **entry)
            return outputs

        worker.rollout.inference_engine.generate = measured_generate
        for task in args.tasks:
            cfg.env.env_name = task
            pair = []
            for name, cls in [('author', original.TrajectoryCollector),
                              ('candidate', rollout_loop.TrajectoryCollector)]:
                cfg.env.max_steps = 15
                train_env, val_env = env_manager.make_envs(cfg)
                val_env.envs.close()
                # This bounds only collection for the fixture. No fake terminal
                # reward is introduced, and task dynamics retain their horizon.
                cfg.env.max_steps = args.rounds
                # Sokoban's owning reset samples from numpy; WebShop owns a
                # seeded RandomState. Recreate the owners and reset this seed.
                np.random.seed(2026)
                initial = DataProto.from_single_dict(dict(
                    input_ids=torch.zeros((4, 1), dtype=torch.long),
                    raw_prompt=np.array([[dict(role='user', content='')]]*4, dtype=object),
                    data_source=np.array(['text']*4, dtype=object)))
                initial.meta_info = {'do_sample': False}
                first_call = len(engine_calls)
                tick = time.perf_counter()
                record('rollout_start', task=task, implementation=name)
                try:
                    # Call the owning interaction loop directly. Credit
                    # readout and PPO live above this boundary and are not run.
                    out = cls(cfg, worker.tokenizer).vanilla_multi_turn_loop(initial, worker, train_env)
                finally:
                    train_env.envs.close()
                    # Let the owner's destructors run while Ray is still live.
                    del train_env, val_env
                rows = trace_rows(out[0])
                pair.append(rows)
                artifact = args.output.with_name(f'{args.output.stem}-{task}-{name}.json')
                artifact.write_text(json.dumps(rows)+'\n')
                run = dict(task=task, implementation=name, seconds=time.perf_counter()-tick,
                           rewards=out[1].tolist(), lengths=out[2].tolist(),
                           success={key: np.asarray(value).tolist() for key, value in out[3].items()},
                           artifact=str(artifact), calls=engine_calls[first_call:],
                           active_prompt_tokens=sum(len(row['prompt']) for ep in rows for row in ep),
                           max_prompt_tokens=max(len(row['prompt']) for ep in rows for row in ep))
                result['runs'].append(run)
                record('rollout_complete', **{k: v for k, v in run.items() if k != 'calls'})
            result.setdefault('comparisons', {})[task] = dict(
                exact_active_prompt_response_reward_match=pair[0] == pair[1],
                exact_return_length_success_match=all(result['runs'][-2][key] == result['runs'][-1][key]
                                                     for key in ('rewards', 'lengths', 'success')))
            record('comparison', task=task, **result['comparisons'][task])
        result['status'] = ('passed_fixed_policy_regression' if all(
            all(v.values()) for v in result['comparisons'].values()) else 'mismatch_requires_inspection')
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['elapsed'] = time.perf_counter()-started
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if ray.is_initialized():
            ray.shutdown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
