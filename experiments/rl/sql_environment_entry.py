"""SkyRL's standalone SQL environment at the existing VERL boundary.

SQLEnv owns parsing, tools, rewards, history messages and turn termination.
The transport buffer only joins original generated IDs and raw observation IDs,
as selected by the pinned recipe's use_conversation_multi_turn=false. Neither
SkyRL's trainer nor its inference client is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np

from owner_environment_transport import PolicyReply, policy_reply


def configure_sql_tokenizer(tokenizer, template_path=None):
    if template_path:
        from pathlib import Path
        # Pinned SkyRL's original Qwen3 template leaves <think> in the action,
        # where its original format reward expects it; no generated IDs change.
        tokenizer.chat_template = Path(template_path).read_text()


@dataclass
class SQLSession:
    env: object
    tokenizer: object
    prompt_ids: list[int]
    max_input_length: int
    done: bool = False

    @classmethod
    def create(cls, tokenizer, db_path, task, max_turns, max_input_length):
        from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig
        env = SQLEnv(Text2SQLEnvConfig(db_path=db_path),
                     dict(task['extras'], max_turns=max_turns))
        messages, _ = env.init(deepcopy(task['prompt']))
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_dict=False)
        return cls(env, tokenizer, ids, max_input_length)

    def step(self, reply: PolicyReply):
        result = self.env.step(reply.text)
        # Exact token-in/token-out single-assistant continuation of the recipe.
        response = reply.token_ids
        if response and response[-1] == self.tokenizer.eos_token_id:
            response = response[:-1]
        self.prompt_ids.extend(response)
        for message in result['observations']:
            self.prompt_ids.extend(self.tokenizer.encode(message['content'], add_special_tokens=False))
        # The owner's generator checks this before its next inference request.
        self.done = result['done'] or len(self.prompt_ids) > self.max_input_length
        return result


def make_sql_environments(configuration, tokenizer):
    from agent_system.environments.base import EnvironmentManagerBase
    from omegaconf import OmegaConf
    config = configuration
    configure_sql_tokenizer(tokenizer, config.data.get('sql_chat_template'))

    class SQLManager(EnvironmentManagerBase):
        def __init__(self, evaluation):
            self.config = config
            self.tokenizer = tokenizer
            self.sessions = []
            self.evaluation = evaluation

        def reset(self, kwargs):
            self.close()
            self.sessions = [SQLSession.create(
                tokenizer, config.env.sql.db_path, task,
                config.env.max_steps, config.env.sql.max_input_length) for task in kwargs]
            return self.observations(), [{} for _ in self.sessions]

        def observations(self):
            options = config.env.sql.eval_sampling if self.evaluation else config.env.sql.sampling
            options = OmegaConf.to_container(options, resolve=True)
            # SkyRL's native SamplingParams leaves detokenize at its default
            # True and SQLEnv consumes RequestOutput.text. VERL normally turns
            # it off because its built-in environments decode response IDs.
            options['detokenize'] = True
            return dict(text=[''] * len(self.sessions), image=None, anchor=None,
                        raw_prompt_ids=[[] if s.done else s.prompt_ids.copy() for s in self.sessions],
                        sampling_kwargs=[dict(options) for _ in self.sessions])

        def step_policy_outputs(self, batch, active_masks):
            rewards = np.zeros(len(self.sessions), dtype=np.float32)
            infos = []
            for i, session in enumerate(self.sessions):
                result = None
                if active_masks[i] and not session.done:
                    result = session.step(policy_reply(batch, i))
                    rewards[i] = result['reward']
                infos.append(dict(won=bool(rewards[i] == 1),
                                  owner_metadata=None if result is None else result['metadata']))
            return self.observations(), rewards, np.array([s.done for s in self.sessions]), infos

        def close(self):
            for session in self.sessions:
                session.env.close()
            self.sessions = []

    return SQLManager(False), SQLManager(True)
