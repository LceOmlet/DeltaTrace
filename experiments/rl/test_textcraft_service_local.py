"""Actual pinned TextCraft FastAPI/environment behind the original HTTP client.

TestClient replaces the network socket only. Original client parser, server,
crafting rules, inventory, reward and reset all run. No trainer/model is loaded.
"""
import importlib
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[2] / 'research/temporary/rl_upstream_alignment_20260929/recipe-sources'
OWNER = BASE / 'AgentGym-d014732d9fe39b975c368c03749bfd50950067f6'


@pytest.fixture
def native_service(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.syspath_prepend(str(OWNER / 'agentenv-textcraft'))
    monkeypatch.syspath_prepend(str(OWNER / 'agentenv'))
    monkeypatch.chdir(OWNER / 'agentenv-textcraft')
    module = importlib.import_module('agentenv_textcraft.server')
    with TestClient(module.app) as client:
        yield client


def test_original_http_client_and_service_reset_step_close(native_service, monkeypatch):
    from agentenv.envs.textcraft import TextCraftEnvClient
    import requests

    # Use the original URLs/data; replace only the HTTP network round trip.
    monkeypatch.setattr(requests, 'post', lambda url, **kw: native_service.post(
        url.replace('http://testserver', ''), json=kw.get('json')))
    monkeypatch.setattr(requests, 'get', lambda url, **kw: native_service.get(
        url.replace('http://testserver', '')))
    client = TextCraftEnvClient('http://testserver', data_len=1)
    try:
        result = client.reset(0)
        assert client.observe() == result['observation']
        assert 'Crafting commands:' in client.observe() and 'Goal: craft' in client.observe()
        step = client.step('Thought: inspect.\nAction: inventory')
        assert step.state == 'Inventory: You are not carrying anything.'
        assert step.reward == 0 and not step.done
        invalid = client.step('Action: inventory\nAction: inventory')
        assert invalid.reward == 0 and not invalid.done
        assert 'Only one' in invalid.state
        assert client.observe() == step.state
    finally:
        client.close()


def test_official_crafting_success_reward_and_inventory(native_service):
    from agentenv_textcraft.env_wrapper import server
    created = server.create()
    index = created['id']
    try:
        environment = server.env[index]
        environment.reset(commands='craft 1 blue dye using 1 lapis lazuli', goal='minecraft:blue_dye')
        first = server.step(index, 'get 1 lapis lazuli')
        assert first['reward'] == 0 and not first['done']
        final = server.step(index, 'craft 1 blue dye using 1 lapis lazuli')
        assert final['reward'] == 1 and final['done'], final
        assert environment.inventory == {'minecraft:blue_dye': 1}
    finally:
        server.close(index)


def test_native_prompt_renderer_and_environment_transport(native_service, monkeypatch):
    from types import ModuleType
    from agentenv.envs.textcraft import TextCraftEnvClient
    from transformers import AutoTokenizer
    import requests
    from patch_agentgym_model_entry import patch
    from textcraft_environment_entry import TextCraftSession
    from owner_environment_transport import PolicyReply
    from test_qwen35_environment_entry_local import ASSETS

    path = BASE / 'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL/verl/workers/rollout/schemas.py'
    schemas = ModuleType('native_agentgym_test')
    exec(compile(patch(path.read_text()), str(path), 'exec'), schemas.__dict__)
    tokenizer = AutoTokenizer.from_pretrained(ASSETS, local_files_only=True)
    monkeypatch.setattr(requests, 'post', lambda url, **kw: native_service.post(
        url.replace('http://testserver', ''), json=kw.get('json')))
    client = TextCraftEnvClient('http://testserver', data_len=1)
    try:
        session = TextCraftSession(client, schemas, tokenizer, 0)
        expected = [dict(role='user', content=client.conversation_start[0]['value']),
                    dict(role='assistant', content=client.conversation_start[1]['value']),
                    dict(role='user', content=client.observe())]
        assert session.prompt_ids() == tokenizer.apply_chat_template(expected,
            add_generation_prompt=True, tokenize=True, return_dict=False)
        text = 'Thought: inspect.\nAction: inventory'
        ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        output = session.step(PolicyReply(ids, [-.1]*len(ids), '', 'stop'))
        expected += [dict(role='assistant', content=text), dict(role='user', content=output.state)]
        assert session.prompt_ids() == tokenizer.apply_chat_template(expected,
            add_generation_prompt=True, tokenize=True, return_dict=False)
        assert output.reward == 0 and not output.done
        assert ids[-1] == tokenizer.eos_token_id  # No mutation of policy artifact.
    finally:
        client.close()
