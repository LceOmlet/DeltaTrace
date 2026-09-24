"""Compare actual task renderers with the author-owned memory prompts.

PINNED_ENV_MANAGER_SOURCE is the unmodified file from upstream.lock's commit.
The explicit marker observations exercise rendering, not task performance.
"""
import ast
import os
from pathlib import Path
from types import MethodType

import pytest
from omegaconf import OmegaConf
from agent_system.environments import env_manager as owner


@pytest.fixture
def pristine():
    return ast.parse(Path(os.environ['PINNED_ENV_MANAGER_SOURCE']).read_text())


def manager(name):
    instance = getattr(owner, name).__new__(getattr(owner, name))
    instance.config = OmegaConf.create({'env': {'history_length': 2}})
    instance.tasks = ['TASK_MARKER']
    instance.is_multi_modal = False
    instance.supervisors = [dict(first_name='Test', last_name='Person', email='fixture@example.com', phone_number='000')]
    instance.memory = owner.SimpleMemory()
    instance.memory.reset(batch_size=1)
    instance.memory.store({'text_obs': ['OLD_OBSERVATION_MARKER'], 'action': ['OLD_ACTION_MARKER']})
    return instance


@pytest.mark.parametrize('name', ['SokobanEnvironmentManager', 'WebshopEnvironmentManager', 'AppWorldEnvironmentManager'])
def test_default_and_initial_prompts_match_pinned_owner(pristine, name):
    instance = manager(name)
    original_class = next(n for n in pristine.body if isinstance(n, ast.ClassDef) and n.name == name)
    fn = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == 'build_text_obs')
    ns = dict(vars(owner))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-owner-build_text_obs>', 'exec'), ns)
    original = MethodType(ns['build_text_obs'], instance)
    observations = ['CURRENT_OBSERVATION_MARKER']
    infos = [{'available_actions': {'has_search_bar': False, 'clickables': ['Blue', 'Buy Now']}}]
    kwargs = {'text_obs': observations}
    if name != 'AppWorldEnvironmentManager':
        kwargs['infos'] = infos
    for init in [False, True]:
        assert instance.build_text_obs(**kwargs, init=init) == original(**kwargs, init=init)
    # A stale full-chat flag must no longer override the author's renderer.
    instance.config.env.full_chat_observations = True
    for init in [False, True]:
        assert instance.build_text_obs(**kwargs, init=init) == original(**kwargs, init=init)
