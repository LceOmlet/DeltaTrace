"""Native AgentGym TextCraft client and prompt renderer in VERL's collector.

Only the two leaf owner modules are loaded by file, avoiding the other project's
same-named ``verl`` package. No AgentGym trainer, engine or optimizer is imported.
Messages passed to the original renderer are the client/step API's exact text.
Original generated IDs stay in VERL; they are never reconstructed from messages.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def owner_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_textcraft_tokenizer(tokenizer):
    from functools import partial
    # Native TextCraft spends its 512-token action on Thought/Action. Qwen3.5
    # otherwise starts a separate, often unfinished reasoning channel. Bind
    # the model's official template option; the owner renderer remains intact.
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template,
                                            enable_thinking=False)


class TextCraftSession:
    def __init__(self, client, schemas, tokenizer, item_id):
        self.client, self.schemas, self.tokenizer = client, schemas, tokenizer
        self.history = SimpleNamespace(messages=[
            schemas.Message('user', client.conversation_start[0]['value']),
            schemas.Message('assistant', client.conversation_start[1]['value'])])
        client.reset(item_id)
        self.history.messages.append(schemas.Message('user', client.observe()))
        self.done = False

    def prompt_ids(self):
        return self.schemas.RolloutHandler.get_generation_prompt(self.history, self.tokenizer)

    def step(self, reply):
        # Native AgentGym's environment interface consumes skip-special text.
        text = self.tokenizer.decode(reply.token_ids, skip_special_tokens=True)
        self.history.messages.append(self.schemas.Message('assistant', text))
        output = self.client.step(text)
        self.history.messages.append(self.schemas.Message('user', output.state))
        self.done = output.done
        return output


def make_textcraft_environments(configuration, tokenizer):
    import os
    import site
    if os.environ.get('TEXTCRAFT_EXTRAS'):
        site.addsitedir(os.environ['TEXTCRAFT_EXTRAS'])
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from agent_system.environments.base import EnvironmentManagerBase
    from owner_environment_transport import policy_reply
    from omegaconf import OmegaConf
    config = configuration
    configure_textcraft_tokenizer(tokenizer)

    root = Path(config.env.textcraft.owner_root) / 'AgentGym-RL'
    schemas = owner_module(root / 'verl/workers/rollout/schemas.py', 'owner_agentgym_schemas')
    client_module = owner_module(root / 'verl/utils/agentgym/client.py', 'owner_agentgym_client')

    class TextCraftManager(EnvironmentManagerBase):
        def __init__(self, evaluation):
            self.config, self.evaluation = config, evaluation
            self.sessions = []
            self.pool = ThreadPoolExecutor()

        def reset(self, kwargs):
            for session in self.sessions:
                session.client.close()
            options = (config.env.textcraft.eval_client if self.evaluation
                       else config.env.textcraft.client)
            self.sessions = list(self.pool.map(lambda task: TextCraftSession(
                client_module.init_env_client(options), schemas, tokenizer, task['item_id']), kwargs))
            return self.observations(), [{} for _ in self.sessions]

        def observations(self):
            options = config.env.textcraft.eval_sampling if self.evaluation else config.env.textcraft.sampling
            return dict(text=['']*len(self.sessions), image=None, anchor=None,
                raw_prompt_ids=[[] if s.done else s.prompt_ids() for s in self.sessions],
                sampling_kwargs=[OmegaConf.to_container(options, resolve=True) for _ in self.sessions])

        def step_policy_outputs(self, batch, active_masks):
            active = [i for i, live in enumerate(active_masks) if live]
            results = self.pool.map(lambda i: self.sessions[i].step(policy_reply(batch, i)), active)
            rewards = np.zeros(len(self.sessions), dtype=np.float32)
            for i, result in zip(active, results):
                rewards[i] = result.reward
            infos = [dict(won=bool(r == 1)) for r in rewards]
            return self.observations(), rewards, np.array([s.done for s in self.sessions]), infos

        def close(self):
            for session in self.sessions:
                session.client.close()
            self.sessions = []
            self.pool.shutdown(wait=True)

    return TextCraftManager(False), TextCraftManager(True)
