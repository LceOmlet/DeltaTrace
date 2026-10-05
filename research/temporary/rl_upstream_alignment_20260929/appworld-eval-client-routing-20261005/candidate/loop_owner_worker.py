"""CPU transport for the original LOOP sampler, runner pool and collector.

These processes own environments only. No model, optimizer or inference server
is constructed. The original get_rollouts performs distributed completion and
cancellation; completion requests are passed to the existing VERL workers.
"""
from queue import Queue, Empty
from threading import Thread, Lock, local
from pathlib import Path
import os
import site
import traceback
import uuid
from functools import partial


_recording = local()


def recorded_runner(**kwargs):
    from phi_agents.rl.appworld_scenario_runner import AppWorldScenarioRunner

    class RecordedRunner(AppWorldScenarioRunner):
        def __init__(self, **options):
            super().__init__(**options)
            execute = self.world.execute
            def recorded_execute(*args, **kwargs):
                _recording.executions += 1
                return execute(*args, **kwargs)
            self.world.execute = recorded_execute
            restart = self.world.restart
            def recorded_restart(*args, **kwargs):
                result = restart(*args, **kwargs)
                # The author discards the failed episode after a successful
                # restart. Its completion records must follow the same scope.
                _recording.requests = []
                _recording.executions = 0
                return result
            self.world.restart = recorded_restart

        def run(self, scenario, llm):
            _recording.requests = []
            _recording.executions = 0
            result = super().run(scenario, llm)
            result.completion_requests = _recording.requests
            result.tool_execution_count = _recording.executions
            return result

    return RecordedRunner(**kwargs)


def run_owner(rank, world_size, init_method, config, model_path, reserve,
              evaluation, commands, replies, output, cancellation, inference_clients=1):
    if os.environ.get('LOOP_EXTRAS'):
        site.addsitedir(os.environ['LOOP_EXTRAS'])
    import torch
    import torch.distributed as dist
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
    from phi_agents.rl.vllm_rollout_worker import VLLMRolloutWorker
    from phi_agents.rl.llm.qwen_3 import VLLMQwen3

    worker = sampler = None
    waiting, mutex = {}, Lock()

    def receive_replies():
        while True:
            value = replies.get()
            if value is None:
                return
            key, reply = value
            with mutex:
                if key in waiting:
                    waiting[key].put(reply)

    Thread(target=receive_replies, daemon=True).start()
    try:
        torch.set_num_threads(1)
        if world_size > 1:
            dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=world_size)
        cfg = OmegaConf.create(config)
        sampler_cfg = cfg.rl.eval.scenario_sampler if evaluation else cfg.rl.scenario_sampler
        sampler = ParallelScenarioSampler(lambda: instantiate(sampler_cfg), num_threads=1)
        client_config = OmegaConf.to_container(cfg.llm.vllm_class, resolve=True)
        client_config.pop('_target_')
        template = VLLMQwen3(host='127.0.0.1', port=0, base_model_path=Path(model_path),
            model_id=None, temperature=cfg.llm.temperature,
            max_model_len=cfg.llm.vllm_server.max_model_len-reserve, **client_config)

        def completion(request, *, server_index=None):
            key, answer = uuid.uuid4().hex, Queue()
            with mutex:
                waiting[key] = answer
            event = ('completion', rank, key, request)
            if server_index is not None:
                event += (server_index,)
            output.put(event)
            try:
                while not cancellation.is_set():
                    try:
                        reply = answer.get(timeout=.1)
                        if isinstance(reply, BaseException):
                            raise reply
                        _recording.requests.append((key, len(request['prompt']), len(reply.token_ids)))
                        stopped = template._vllm._is_max_tokens_stopped(dict(finish_reason=reply.finish_reason))
                        return reply.text, reply.token_ids, reply.logprobs, stopped, False
                    except Empty:
                        pass
                return '', [], [], False, True
            finally:
                with mutex:
                    waiting.pop(key, None)

        template._vllm._completion_transport = completion
        eval_clients = [template]
        if evaluation:
            if inference_clients < 1:
                raise ValueError('Evaluation requires at least one native server')
            for _ in range(1, inference_clients):
                eval_clients.append(VLLMQwen3(host='127.0.0.1', port=0, base_model_path=Path(model_path),
                    model_id=None, temperature=cfg.llm.temperature,
                    max_model_len=cfg.llm.vllm_server.max_model_len-reserve, **client_config))
            for server_index, client in enumerate(eval_clients):
                client._vllm._completion_transport = partial(completion, server_index=server_index)
        runner_cfg = OmegaConf.to_container(cfg.rl.scenario_runner, resolve=True)
        runner_cfg['_target_'] = 'loop_owner_worker.recorded_runner'
        def clients(event):
            if evaluation:
                for client in eval_clients:
                    client._vllm._cancellation_event = event
                return eval_clients
            template._vllm._cancellation_event = event
            return [template]
        worker = VLLMRolloutWorker(llm_cfg=cfg.llm, scenario_sampler=sampler,
            rollouts_per_scenario=cfg.rl.params.rollouts_per_scenario,
            runner_cfg=OmegaConf.create(runner_cfg), rank=rank, local_rank=rank,
            barrier=dist.barrier, inference_gpus=[0], exclusive_inference_and_learning=False,
            max_gpu_mem_utilization=None, num_runners=cfg.rl.num_scenario_runners,
            external_llm_factory=clients, external_cancellation_event=cancellation)
        output.put(('ready', rank))
        while True:
            command = commands.get()
            if command == 'stop':
                break
            count = cfg.rl.params.scenarios_per_iteration // world_size
            worker.request_rollout_generation(count, None)
            options = {} if evaluation else dict(rollouts_fraction=cfg.rl.rollouts_fraction,
                rollouts_per_scenario_fraction=cfg.rl.rollouts_per_scenario_fraction)
            result = worker.get_rollouts(count, world_size=world_size,
                device=torch.device('cpu'), **options)
            output.put(('result', rank, result))
    except BaseException:
        output.put(('error', rank, traceback.format_exc()))
    finally:
        if worker is not None:
            worker.stop()
        if sampler is not None:
            sampler.stop()
        if dist.is_initialized():
            dist.destroy_process_group()


class OwnerProcesses:
    """IPC only; task selection and scheduling stay inside VLLMRolloutWorker."""
    def __init__(self, config, model_path, reserve, evaluation, world_size, inference_clients=1):
        import multiprocessing as mp
        import tempfile
        context = mp.get_context('spawn')
        self.output = context.Queue()
        self.commands = [context.Queue() for _ in range(world_size)]
        self.replies = [context.Queue() for _ in range(world_size)]
        self.cancellations = [context.Event() for _ in range(world_size)]
        self.directory = tempfile.TemporaryDirectory(prefix='dt-loop-')
        init = Path(self.directory.name, 'gloo').as_uri()
        self.processes = [context.Process(target=run_owner, args=(rank, world_size, init,
            config, model_path, reserve, evaluation, self.commands[rank], self.replies[rank],
            self.output, self.cancellations[rank], inference_clients)) for rank in range(world_size)]
        for process in self.processes:
            process.start()
        self.ready = False

    def event(self, block=True):
        while True:
            try:
                value = self.output.get(timeout=1) if block else self.output.get_nowait()
                if value[0] == 'error':
                    raise RuntimeError(value[2])
                return value
            except Empty:
                failed = [(p.pid, p.exitcode) for p in self.processes if p.exitcode is not None]
                if failed:
                    raise RuntimeError(f'LOOP environment process exited: {failed}')
                if not block:
                    raise

    def queued_event(self):
        # multiprocessing.Queue counts a put before its feeder has written the
        # object to the pipe. get_nowait can therefore report Empty while many
        # accepted requests are queued. There is one consumer: wait for an
        # already-counted item, without waiting for a new environment request.
        if self.output.qsize() == 0:
            raise Empty
        return self.event()

    def start(self):
        if not self.ready:
            for _ in self.processes:
                event = self.event()
                assert event[0] == 'ready', event
            self.ready = True
        for command in self.commands:
            command.put('collect')

    def close(self):
        for event, command in zip(self.cancellations, self.commands):
            event.set()
            command.put('stop')
        for process in self.processes:
            process.join(timeout=30)
        live = [p.pid for p in self.processes if p.is_alive()]
        if live:
            raise RuntimeError(f'LOOP cleanup still running: {live}')
        self.directory.cleanup()
