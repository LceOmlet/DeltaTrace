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
    from loop_owner_rollout import LoopOwner
    return LoopOwner(configuration, tokenizer, False), LoopOwner(configuration, tokenizer, True)
