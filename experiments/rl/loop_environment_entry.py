"""LOOP's native AppWorld episode, with completion transport supplied by VERL.

The original runner/agent/LLM client retains the complete message loop, parsing,
budget termination, evaluation, fractional train reward and sparse eval reward.
Only its normalized completion HTTP boundary uses a rendezvous carrying exact
token IDs to the existing VERL vLLM worker. No LOOP trainer/server is launched.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Thread

from owner_environment_transport import PolicyReply


@dataclass(frozen=True)
class CompletionRequest:
    prompt_ids: list[int]
    sampling_kwargs: dict


@dataclass(frozen=True)
class EpisodeResult:
    rollout: object = None
    error: BaseException | None = None


def training_readout_reserve(native, tokenizer):
    """Reserve the existing DT query inside the native context budget."""
    import json
    import os
    from pathlib import Path
    from phi_agents.appworld.interface import load_task_ids
    from reward_readout import RewardAlphabet

    split = native['training_task_sampler']['dataset_name']
    tasks = Path(os.environ['APPWORLD_ROOT']) / 'data/tasks'
    counts = {len(json.loads((tasks / task / 'ground_truth/test_data.json').read_text()))
              for task in load_task_ids(split)}
    steps = native['training_environment']['appworld_config']['env']['max_interactions']
    sampling = json.loads(os.environ['DT_SAMPLING_JSON'])
    return max(RewardAlphabet.for_task('AppWorld', steps, appworld_num_tests=n)
               .readout_token_budget(tokenizer, steps, sampling) for n in counts)


class LoopEpisode:
    """Synchronous owner calls rendezvous with VERL's existing batched collector."""

    def __init__(self, runner, scenario, llm):
        self.requests = Queue()
        self.replies = Queue()
        self.runner = runner
        self.scenario = scenario
        self.llm = llm
        # This is the sole replaced owner call; no inference server is created.
        llm._vllm._completion_transport = self.get_completion
        self.thread = Thread(target=self._run, name='loop-environment', daemon=True)

    def _run(self):
        try:
            result = EpisodeResult(rollout=self.runner.run(self.scenario, self.llm))
        except BaseException as error:
            result = EpisodeResult(error=error)
        self.requests.put(result)

    def get_completion(self, request):
        # The original LOOP client has already normalized optional parameters.
        # Only HTTP envelope fields are removed; VERL owns the loaded policy.
        options = dict(request)
        prompt_tokens = options.pop('prompt')
        options.pop('model')
        options.pop('stream')
        options['detokenize'] = True  # The native agent consumes completion text.
        self.requests.put(CompletionRequest(list(prompt_tokens), options))
        reply = self.replies.get()
        if isinstance(reply, BaseException):
            raise reply
        stopped = self.llm._vllm._is_max_tokens_stopped(dict(finish_reason=reply.finish_reason))
        return reply.text, reply.token_ids, reply.logprobs, stopped, False

    def start(self):
        self.thread.start()
        return self.next_event()

    def next_event(self):
        event = self.requests.get()
        if isinstance(event, EpisodeResult) and event.error is not None:
            raise event.error
        return event

    def advance(self, reply: PolicyReply):
        self.replies.put(reply)
        return self.next_event()

    def cancel(self):
        # Wake an owner blocked at the completion boundary. Owner's finally
        # handles world cleanup; the runner's server lifecycle remains explicit.
        if self.thread.is_alive():
            self.replies.put(InterruptedError('VERL environment closed'))
            self.thread.join(timeout=10)


def make_loop_environments(configuration, tokenizer):
    import os
    import site
    if os.environ.get('LOOP_EXTRAS'):
        site.addsitedir(os.environ['LOOP_EXTRAS'])
    from concurrent.futures import ThreadPoolExecutor
    from copy import copy
    from pathlib import Path
    import numpy as np
    from hydra.utils import instantiate
    from agent_system.environments.base import EnvironmentManagerBase
    from phi_agents.rl.appworld_scenario_runner import AppWorldScenario
    from phi_agents.rl.llm.qwen_3 import VLLMQwen3
    from loop_owner_recipe import environment_configuration
    from owner_environment_transport import policy_reply
    config = configuration

    native = environment_configuration(Path(config.env.loop.owner_root))
    boundary = native['generation_boundary_reference']
    reserve = (training_readout_reserve(native, tokenizer)
               if config.get('algorithm', {}).get('adv_estimator') == 'deltatrace' else 0)

    class LoopManager(EnvironmentManagerBase):
        def __init__(self, phase):
            self.config, self.phase = config, phase
            self.runners, self.episodes, self.events = [], [], []
            self.pool = ThreadPoolExecutor()
            client = dict(boundary['client'])
            client.pop('_target_')
            # Configure the native client's existing length/termination logic.
            # Evaluation has no DT readout. No history is truncated here.
            max_length = boundary['max_model_len'] - (reserve if phase == 'training' else 0)
            # Existing, tokenizer-compatible native Qwen3 client. It owns message
            # encoding and decoding; the supplied tokenizer has identical IDs.
            self.template = VLLMQwen3(
                host='127.0.0.1', port=0, base_model_path=Path(config.actor_rollout_ref.model.path),
                model_id=None, temperature=boundary[f'{phase}_temperature'],
                max_model_len=max_length, **client)
            self.template._tokenizer = tokenizer

        def reset(self, kwargs):
            for episode in self.episodes:
                episode.cancel()
            if len(self.runners) != len(kwargs):
                for runner in self.runners:
                    runner.cleanup()
                self.runners = list(self.pool.map(
                    lambda _: instantiate(native[f'{self.phase}_environment'], _recursive_=False), kwargs))
            self.episodes = []
            for runner, task in zip(self.runners, kwargs):
                llm = copy(self.template)
                llm._vllm = copy(self.template._vllm)
                scenario = AppWorldScenario(task_id=task['task_id'],
                    dataset_name=native[f'{self.phase}_task_sampler']['dataset_name'])
                self.episodes.append(LoopEpisode(runner, scenario, llm))
            self.events = list(self.pool.map(lambda e: e.start(), self.episodes))
            return self.observations(), [{} for _ in self.episodes]

        def observations(self):
            return dict(text=['']*len(self.events), image=None, anchor=None,
                raw_prompt_ids=[e.prompt_ids if isinstance(e, CompletionRequest) else [] for e in self.events],
                sampling_kwargs=[e.sampling_kwargs if isinstance(e, CompletionRequest) else {} for e in self.events])

        def step_policy_outputs(self, batch, active_masks):
            live = [i for i, active in enumerate(active_masks) if active]
            updated = self.pool.map(lambda i: self.episodes[i].advance(policy_reply(batch, i)), live)
            rewards = np.zeros(len(self.events), dtype=np.float32)
            for i, event in zip(live, updated):
                self.events[i] = event
                if isinstance(event, EpisodeResult):
                    rewards[i] = event.rollout.ret
            infos = [dict(won=bool(e.rollout.appworld_rollout_data.eval_result.success))
                     if isinstance(e, EpisodeResult) else dict(won=False) for e in self.events]
            return self.observations(), rewards, np.array([isinstance(e, EpisodeResult) for e in self.events]), infos

        def finalize_trajectory_metadata(self, total_batch_list):
            # Native terminal metadata is copied to its own earlier response
            # rows. This passes the categorical denominator, not a new reward.
            for rows, event in zip(total_batch_list, self.events):
                if not isinstance(event, EpisodeResult):
                    raise RuntimeError('Collector ended before the native LOOP episode completed')
                count = event.rollout.appworld_rollout_data.eval_result.num_tests
                for row in rows:
                    row['appworld_num_tests'] = count

        def close(self):
            for episode in self.episodes:
                episode.cancel()
            for runner in self.runners:
                runner.cleanup()
            self.pool.shutdown(wait=True)

    return LoopManager('training'), LoopManager('evaluation')
