"""Full original TextCraft episodes before any DT or optimizer update.

The native dataset, TextCraftOwner, AgentGym rollout/handler/client and VERL
actor->vLLM RPC own all execution. Hooks persist their actual requests/results.
Controlled cases only supply explicit SamplingParams seeds and compose the
existing select_idxs/RPC interfaces to match eight replica submission groups.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType
import time
import traceback

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RUN = ROOT / 'runs/textcraft-dtpo-v5-original-2gpu-formal-20261021-v1'
AUDIT = ROOT / 'receipts/textcraft-generation-numerical-20261021-v1'
PREPARED = AUDIT / 'full-first-rollout-prepared'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_seed(base, item, sample, turn):
    # Fixed before observing scores; independent of topology and completion order.
    identity = json.dumps([int(base), int(item), int(sample), int(turn)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63 - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=(2, 8), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=('native', 'coupled', 'matched'), required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(AUDIT)
    args.output.mkdir(exist_ok=False)

    import numpy as np
    import psutil
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils import hf_tokenizer
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from textcraft_environment_entry import make_textcraft_environments
    from inspect_native_initial_generation import lora_tensor_record

    source = json.loads((RUN / 'source.json').read_text())
    config = OmegaConf.load(PREPARED / 'configuration.yaml')
    config.actor_rollout_ref.actor.optim.total_training_steps = 330
    # Only return the native rollout/reward tensors. No advantage computation is called.
    config.algorithm.adv_estimator = 'grpo'
    expected_worker = Path(source['verl_root']) / 'verl/workers/fsdp_workers.py'
    assert Path(inspect.getfile(ActorRolloutRefWorker)).resolve() == expected_worker.resolve()
    assert sha(expected_worker) == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
    preparation = json.loads((PREPARED / 'preparation.json').read_text())
    assert sha(preparation['batch_path']) == preparation['batch_SHA256']

    result = dict(status='initializing_native_zero_update_full_rollout', started_unix=time.time(),
                  pid=os.getpid(), birth=psutil.Process().create_time(), world_size=args.world_size,
                  physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'], source_run=str(RUN),
                  prepared_batch_SHA256=preparation['batch_SHA256'], script_SHA256=sha(__file__),
                  workload=dict(groups=32, samples_per_group=8, full_episodes=256,
                                max_rounds=int(config.env.max_steps), native_action_token_limit=512),
                  cases={}, DT_calls=0, optimizer_updates=0, formal_training_started=False,
                  tolerance_verdict=None, historical_first_score_cause_proven=False)

    def save():
        (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')

    @ray.remote
    class ObservedWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def observe_once(self):
            self.observation_case = None
            self.observation_call = 0
            self.observation_rank_dir = args.output / f'rank{self.rank}'
            self.observation_rank_dir.mkdir(exist_ok=False)
            manager = self.rollout_sharding_manager
            original_update = manager.update_params
            original_generate = self.rollout.inference_engine.generate
            sync_observations = []

            def observed_update(instance, params, peft_config=None):
                answer = original_update(params, peft_config=peft_config)
                if not sync_observations:
                    adapter_owner = getattr(instance.model_runner.lora_manager, '_adapter_manager', None)
                    adapters = getattr(adapter_owner, '_registered_adapters', {})
                    sync_observations.append(dict(rank=self.rank, pid=os.getpid(),
                        birth=psutil.Process().create_time(), base_sync_done=manager.base_sync_done,
                        transferred_lora_B={key: lora_tensor_record(value) for key, value in params.items()
                                            if 'lora_B' in key},
                        actual_registered_lora_B={str(identity): {
                            name: lora_tensor_record(weights.lora_b) for name, weights in adapter.loras.items()}
                            for identity, adapter in adapters.items()}))
                    (self.observation_rank_dir / 'initial-sync.json').write_text(json.dumps(sync_observations, indent=2))
                return answer

            def observed_generate(engine, *positional, **keywords):
                directory = self.observation_rank_dir / self.observation_case
                directory.mkdir(exist_ok=True)
                index = self.observation_call
                self.observation_call += 1
                # Exact owner arguments and returned vLLM objects, including enum values.
                torch.save(dict(prompts=keywords['prompts'], sampling_params=keywords['sampling_params']),
                           directory / f'{index:03d}-requests.pt')
                answer = original_generate(*positional, **keywords)
                torch.save(answer, directory / f'{index:03d}-outputs.pt')
                return answer

            manager.update_params = MethodType(observed_update, manager)
            self.rollout.inference_engine.generate = MethodType(observed_generate, self.rollout.inference_engine)
            return dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
                        worker_path=str(expected_worker), worker_SHA256=sha(expected_worker),
                        optimizer_state_entries=len(self.actor_optimizer.state))

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def set_case(self, name):
            self.observation_case = name
            self.observation_call = 0
            return dict(rank=self.rank, optimizer_state_entries=len(self.actor_optimizer.state))

    environment = {key: value for key, value in os.environ.items()
                   if key.startswith(('DT_', 'VERL_', 'MACA_')) or key in
                   ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH', 'VENV_PYTHON', 'MODEL_PATH',
                    'HF_HUB_OFFLINE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')}
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    environment['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(args.output / 'worker-runtime-overrides')
    save()
    ray.init(address='local', num_cpus=16, num_gpus=args.world_size, include_dashboard=False,
             _temp_dir=f'/tmp/fr-{os.getpid()}',
             runtime_env=dict(env_vars=environment, worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
    try:
        group = RayWorkerGroup(RayResourcePool([args.world_size], use_gpu=True, max_colocate_count=1),
                               RayClassWithInitArgs(ObservedWorker, config.actor_rollout_ref, 'actor_rollout'))
        group.init_model()
        result['workers'] = group.observe_once()
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
        train_owner, eval_owner = make_textcraft_environments(config, tokenizer)
        assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == preparation['template_after_SHA256']
        result['textcraft_owner_path'] = inspect.getfile(type(train_owner))
        result['textcraft_owner_SHA256'] = sha(result['textcraft_owner_path'])
        result['effective_template_SHA256'] = preparation['template_after_SHA256']
        base = DataProto.load_from_disk(preparation['batch_path'])

        for case in args.cases:
            directory = args.output / case
            directory.mkdir(exist_ok=False)
            state = dict(status='running_native_full_rollout', started_unix=time.time(),
                         rng='original_rank_stream' if case == 'native' else 'fixed_task_sample_turn_seed',
                         submission='original_world_size_dispatch' if case != 'matched' else 'eight_replica_groups',
                         completed_native_generation_rounds=0)
            result['cases'][case] = state
            result.update(status='native_full_rollout', phase=case)
            save()
            group.set_case(case)
            handlers, clients, episodes, requests = [], [], [], []
            module = train_owner.module
            original_handler, original_client = module.RolloutHandler, module.init_env_client
            collector = TrajectoryCollector(config, tokenizer)
            original_preprocess = collector.preprocess_batch

            class ObservedHandler(original_handler):
                def __init__(self, *positional, **keywords):
                    super().__init__(*positional, **keywords)
                    self.observation_index = len(handlers)
                    self.observation_requests = 0
                    handlers.append(self)

            def observed_client(*positional, **keywords):
                client = original_client(*positional, **keywords)
                index = len(clients)
                clients.append(client)
                episode = dict(row=index, item_id=None, reset=None, steps=[], errors=[])
                episodes.append(episode)
                reset, step, close = client.reset, client.step, client.close

                def observed_reset(instance, item=0):
                    episode['item_id'] = int(item)
                    answer = reset(item)
                    episode['reset'] = answer
                    return answer

                def observed_step(instance, action):
                    entry = dict(action=action)
                    try:
                        answer = step(action)
                        entry.update(state=answer.state, reward=answer.reward, done=answer.done)
                        return answer
                    except BaseException:
                        episode['errors'].append(traceback.format_exc())
                        raise
                    finally:
                        episode['steps'].append(entry)

                def observed_close(instance):
                    answer = close()
                    episode['closed'] = True
                    return answer

                client.reset = MethodType(observed_reset, client)
                client.step = MethodType(observed_step, client)
                client.close = MethodType(observed_close, client)
                return client

            def observed_preprocess(instance, batch, obs):
                active = [handler for handler in handlers if not handler.done]
                assert len(active) == len(obs['raw_prompt_ids'])
                values = [dict(value) for value in obs['sampling_kwargs']]
                for value, handler in zip(values, active):
                    if case != 'native':
                        value['seed'] = request_seed(config.data.get('seed', 1), handler.item_id,
                                                     handler.observation_index % 8, handler.observation_requests)
                    handler.observation_requests += 1
                exact = dict(obs, sampling_kwargs=values)
                round_index = len(requests)
                torch.save(dict(rows=[h.observation_index for h in active], **exact),
                           directory / f'round{round_index:02d}-actual-owner-requests.pt')
                requests.append(dict(round=round_index, active_rows=[h.observation_index for h in active],
                                     seeds=[v.get('seed') for v in values]))
                return original_preprocess(batch, exact)

            class SubmissionView:
                world_size = group.world_size
                # Used only by the minimal TextCraftOwner padding seam.
                generation_padding_divisor = 8 if case == 'matched' else group.world_size

                def generate_sequences(self, batch):
                    if case != 'matched' or group.world_size == 8:
                        answer = group.generate_sequences(batch)
                    else:
                        assert len(batch) % 8 == 0
                        width = len(batch) // 8 * group.world_size
                        answer = DataProto.concat([
                            group.generate_sequences(batch.select_idxs(np.arange(start, start + width)))
                            for start in range(0, len(batch), width)])
                    state['completed_native_generation_rounds'] += 1
                    state['last_active_requests'] = len(requests[-1]['active_rows'])
                    (directory / 'environment-observations.json').write_text(json.dumps(episodes, indent=2))
                    save()
                    return answer

            module.RolloutHandler, module.init_env_client = ObservedHandler, observed_client
            collector.preprocess_batch = MethodType(observed_preprocess, collector)
            train_owner.native['train']['actor_rollout_ref']['rollout']['rollout_log_dir'] = str(directory / 'native-executer-logs')
            gen_batch = copy.deepcopy(base)
            gen_batch.meta_info['global_steps'] = 1
            try:
                output = train_owner.collect_native_trajectories(collector, gen_batch, SubmissionView(), True)
                output.save_to_disk(directory / 'native-full-rollout.pkl')
                scores = output.non_tensor_batch['episode_rewards'].tolist()
                assert len(scores) == 256
                for row, handler in enumerate(handlers):
                    episodes[row].update(final_score=float(scores[row]), task_rounds=float(output.batch['task_rounds'][row]),
                                         messages=[message.to_dict() for message in handler.messages])
                state.update(status='completed_native_full_rollout', finished_unix=time.time(),
                             score_sum=float(sum(scores)), score_mean=float(sum(scores) / len(scores)),
                             scores=scores, environment_errors=sum(len(e['errors']) for e in episodes),
                             rounds=requests, zero_training_updates=True)
            finally:
                module.RolloutHandler, module.init_env_client = original_handler, original_client
                (directory / 'environment-observations.json').write_text(json.dumps(episodes, indent=2))
                for client, episode in zip(clients, episodes):
                    if not episode.get('closed'):
                        try:
                            client.close()
                        except Exception:
                            episode['close_error'] = traceback.format_exc()
                save()
        train_owner.close()
        eval_owner.close()
        result.update(status='completed_native_full_first_rollout_comparison_cases', finished_unix=time.time())
    except BaseException:
        result.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        save()
        ray.shutdown()


if __name__ == '__main__':
    main()
