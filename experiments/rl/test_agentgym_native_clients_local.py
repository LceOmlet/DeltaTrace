"""Exercise pinned AgentGym clients and RL factory with an HTTP response fixture.

The clients, parsers, retry owner and reward mapping are imported unchanged.
The HTTP fixture is not TextCraft/WebArena and cannot establish task success.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from agentenv.envs.textcraft import TextCraftEnvClient
from agentenv.envs.webarena import WebarenaEnvClient


def response(data):
    return SimpleNamespace(status_code=200, json=lambda: data)


def factory():
    root = Path(__file__).resolve().parents[2]
    path = root / ('research/temporary/rl_upstream_alignment_20260929/recipe-sources/'
                   'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/'
                   'AgentGym-RL/verl/utils/agentgym/client.py')
    spec = importlib.util.spec_from_file_location('official_agentgym_client_factory', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


@pytest.mark.parametrize('task,cls,creation', [
    ('textcraft', TextCraftEnvClient, {'id': 42, 'observation': 'start'}),
    ('webarena', WebarenaEnvClient, {'env_idx': 42}),
])
def test_native_factory_selects_exact_client_and_effective_timeout(monkeypatch, task, cls, creation):
    post = Mock(return_value=response(creation))
    monkeypatch.setattr(requests, 'post', post)
    client = factory().init_env_client(SimpleNamespace(task_name=task, env_addr='http://fixture.invalid',
                                                      timeout=600, max_retries=0))
    assert type(client) is cls
    assert client.timeout == 2400  # Actual owner call, not the nominal script's 600.
    assert post.call_args.kwargs['timeout'] == 2400
    assert post.call_args.args == ('http://fixture.invalid/create',)


def test_textcraft_original_parser_reset_and_fractional_reward(monkeypatch):
    post = Mock(side_effect=[response({'id': 42, 'observation': 'initial'}),
                             response({'observation': 'task prompt'}),
                             response({'observation': 'crafted', 'reward': .375, 'done': True}),
                             response({'closed': True})])
    monkeypatch.setattr(requests, 'post', post)
    client = TextCraftEnvClient('http://fixture.invalid', data_len=1)
    client.reset(17)
    assert post.call_args.kwargs['json'] == {'id': 42, 'data_idx': 17}
    out = client.step('Thought: prepare.\nAction: get  1 lapis-lazuli!')
    assert post.call_args.kwargs['json'] == {'id': 42, 'action': 'get 1 lapislazuli'}
    assert (out.state, out.reward, out.done) == ('crafted', .375, True)
    assert client.observe() == 'crafted'
    client.close()
    assert post.call_args.args == ('http://fixture.invalid/close',)


def test_textcraft_multiple_actions_do_not_reach_server(monkeypatch):
    post = Mock(return_value=response({'id': 42, 'observation': 'initial'}))
    monkeypatch.setattr(requests, 'post', post)
    client = TextCraftEnvClient('http://fixture.invalid', data_len=1)
    out = client.step('Action: inventory\nAction: get 1 stone')
    assert post.call_count == 1 and out.reward == 0 and not out.done


@pytest.mark.parametrize('done', [False, True])
def test_webarena_native_full_action_and_terminal_reward(monkeypatch, done):
    post = Mock(side_effect=[response({'env_idx': 42}), response({'observation': 'task prompt'}),
                             response({'observation': 'page', 'reward': .75, 'terminated': done})])
    monkeypatch.setattr(requests, 'post', post)
    client = WebarenaEnvClient('http://fixture.invalid', data_len=1)
    client.reset(17)
    assert post.call_args.kwargs['json'] == {'env_idx': 42, 'seed': 0, 'idx': 17}
    action = "Let's think. In summary, ```click [42]```"
    out = client.step(action)
    assert post.call_args.kwargs['json'] == {'env_idx': 42, 'action': action}
    assert out.state == 'page' and out.done == done
    assert out.reward == (.75 if done else 0)


def test_webarena_native_parse_failure_and_reset_timeout(monkeypatch):
    post = Mock(side_effect=[response({'env_idx': 42}), response({'observation': 'TimeoutError'})])
    monkeypatch.setattr(requests, 'post', post)
    client = WebarenaEnvClient('http://fixture.invalid', data_len=1)
    out = client.step('click [42]')
    assert post.call_count == 1 and out.reward == 0 and not out.done
    with pytest.raises(TimeoutError, match='item id=17'):
        client.reset(17)
