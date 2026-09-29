"""SkyRL's original SQL agent loop with VERL as its completion transport.

The native generator owns observations, EOS handling, masks, reward positions,
length termination and the final TrajectoryOutput. No SkyRL inference engine or
trainer is constructed. The subclass only records native turn boundaries for
the existing per-response DT interface.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from queue import Queue
from threading import Thread


@dataclass
class Request:
    prompt_ids: list[int]
    sampling_kwargs: dict
    future: object


@dataclass
class Result:
    trajectory: object = None
    error: BaseException | None = None


class SQLRollouts:
    def __init__(self, tokenizer, native, db_path, *, evaluation=False):
        from omegaconf import OmegaConf
        from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
        from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

        self.row = ContextVar('sql_owner_row')
        self.evaluation = evaluation
        self.native = native
        self.events = []
        self.records = []
        self.tasks = []
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, name='sql-owner-loop', daemon=True)
        self.thread.start()
        transport = self

        class RecordedGenerator(SkyRLGymGenerator):
            def _update_agent_loop_state_with_singleturn_chat_template(self, state, turn):
                start = len(state.input_ids)
                state = super()._update_agent_loop_state_with_singleturn_chat_template(state, turn)
                transport.records[transport.row.get()].append({
                    'start': start, 'end': state.response_end_idx + 1,
                    'reward': turn.reward, 'output_ids': turn.output_ids.copy(),
                })
                return state

        cfg = GeneratorConfig.from_dict_config(OmegaConf.create(native['generator']))
        gym_cfg = SkyRLGymConfig()
        gym_cfg.text2sql.db_path = str(db_path)
        self.generator = RecordedGenerator(cfg, gym_cfg, self, tokenizer)

    async def generate(self, input_batch, model=None):
        # This API carries exactly one native episode request. Batched generation
        # still belongs to VERL's worker and the existing official vLLM engine.
        ids = input_batch['prompt_token_ids']
        assert len(ids) == 1
        future = asyncio.get_running_loop().create_future()
        options = dict(input_batch['sampling_params'])
        options['detokenize'] = True
        self.events[self.row.get()].put(Request(ids[0].copy(), options, future))
        return await future

    async def finish_session(self, session_id):
        # VERL uses its existing stateless LLM.generate API, with no external
        # replica reservation to release. Native episode cleanup still runs.
        return None

    async def _episode(self, index, task):
        self.row.set(index)
        try:
            from skyrl.train.generators.base import TrajectoryID
            gen = self.native['generator']
            sampling = self.native['eval_sampling' if self.evaluation else 'sampling']
            trajectory = await self.generator.agent_loop(
                prompt=task['prompt'], env_class='text2sql', env_extras=dict(task['extras']),
                max_tokens=gen['sampling_params']['max_generate_length'],
                max_input_length=gen['max_input_length'], sampling_params=dict(sampling),
                trajectory_id=TrajectoryID(str(index), 0),
                training_phase='eval' if self.evaluation else 'train')
            result = Result(trajectory=trajectory)
        except BaseException as error:
            result = Result(error=error)
        self.events[index].put(result)

    def start(self, tasks):
        self.events = [Queue() for _ in tasks]
        self.records = [[] for _ in tasks]
        self.tasks = [asyncio.run_coroutine_threadsafe(self._episode(i, task), self.loop)
                      for i, task in enumerate(tasks)]
        return [self.next_event(i) for i in range(len(tasks))]

    def next_event(self, index):
        event = self.events[index].get()
        if isinstance(event, Result) and event.error is not None:
            raise event.error
        return event

    def reply(self, request, reply):
        output = dict(responses=[reply.text], response_ids=[reply.token_ids],
                      stop_reasons=[reply.finish_reason], response_logprobs=[reply.logprobs])
        self.loop.call_soon_threadsafe(request.future.set_result, output)

    def close(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        async def finish():
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        asyncio.run_coroutine_threadsafe(finish(), self.loop).result()
        if self.generator.env_executor is not None:
            self.generator.env_executor.shutdown(wait=True)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def make_sql_owner_environments(configuration, tokenizer):
    import json
    from pathlib import Path
    import numpy as np
    from agent_system.environments.base import EnvironmentManagerBase
    from owner_environment_transport import policy_reply
    from sql_environment_entry import configure_sql_tokenizer

    config = configuration
    configure_sql_tokenizer(tokenizer, config.data.get('sql_chat_template'))
    native = json.loads((Path(__file__).with_name('owner_environment_configs.json')).read_text())['SkyRL-SQL']

    class Manager(EnvironmentManagerBase):
        def __init__(self, evaluation):
            self.config = config
            self.rollouts = SQLRollouts(tokenizer, native, config.env.sql.db_path, evaluation=evaluation)
            self.current = []

        def reset(self, kwargs):
            self.current = self.rollouts.start(kwargs)
            return self.observations(), [{} for _ in kwargs]

        def observations(self):
            return dict(text=[''] * len(self.current), image=None, anchor=None,
                raw_prompt_ids=[e.prompt_ids if isinstance(e, Request) else [] for e in self.current],
                sampling_kwargs=[e.sampling_kwargs if isinstance(e, Request) else {} for e in self.current])

        def step_policy_outputs(self, batch, active_masks):
            active = [i for i, value in enumerate(active_masks) if value]
            for i in active:
                self.rollouts.reply(self.current[i], policy_reply(batch, i))
            rewards = np.zeros(len(self.current), dtype=np.float32)
            for i in active:
                self.current[i] = self.rollouts.next_event(i)
                rewards[i] = self.rollouts.records[i][-1]['reward']
            done = np.array([isinstance(e, Result) for e in self.current])
            infos = [dict(won=bool(done[i] and sum(self.current[i].trajectory.reward) == 1.))
                     for i in range(len(self.current))]
            return self.observations(), rewards, done, infos

        def close(self):
            self.rollouts.close()

        def gather_rollout_data(self, collector, **collected):
            from owner_trajectory_batch import sql_trajectory_batch
            return sql_trajectory_batch(self, collector, **collected)

    return Manager(False), Manager(True)
