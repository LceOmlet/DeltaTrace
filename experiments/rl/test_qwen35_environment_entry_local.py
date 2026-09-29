"""Qwen3.5-9B tokenizer/environment entry only; never constructs a model."""
import importlib.util
import ast
import json
from pathlib import Path
from typing import Optional, List
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/temporary/rl_upstream_alignment_20260929'
ASSETS = BASE / 'qwen35-entry-assets'
SOURCES = BASE / 'recipe-sources'


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(ASSETS, local_files_only=True)


def test_official_metadata_and_chat_token_ids(tokenizer):
    cfg = json.loads((ASSETS / 'config.json').read_text())
    assert cfg['architectures'] == ['Qwen3_5ForConditionalGeneration']
    assert cfg['model_type'] == 'qwen3_5'
    assert cfg['text_config']['model_type'] == 'qwen3_5_text'
    assert cfg['text_config']['max_position_embeddings'] >= 32768
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.pad_token_id == 248044
    for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        assert len(encoded) == 1 and tokenizer.decode(encoded) == token


def test_official_transformers_registry_accepts_qwen35_without_loading_weights():
    cfg = AutoConfig.from_pretrained(ASSETS, local_files_only=True)
    assert cfg.model_type == 'qwen3_5'
    assert cfg.text_config.model_type == 'qwen3_5_text'
    assert AutoModelForCausalLM._model_mapping[type(cfg)].__name__ == 'Qwen3_5ForCausalLM'
    assert AutoModelForCausalLM._model_mapping[type(cfg.text_config)].__name__ == 'Qwen3_5ForCausalLM'


@pytest.mark.parametrize('owner', ['verl-agent-20bd331',
    'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL'])
def test_native_framework_tokenizer_entry_preserves_official_chat(owner, tokenizer):
    path = SOURCES / owner / 'verl/utils/tokenizer.py'
    spec = importlib.util.spec_from_file_location('owner_tokenizer_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.hf_tokenizer(str(ASSETS), local_files_only=True)
    messages = [
        {'role': 'system', 'content': 'Use the task environment.'},
        {'role': 'user', 'content': 'Read the inventory.'},
        {'role': 'assistant', 'content': 'Action: inventory'},
        {'role': 'user', 'content': 'inventory: wood, stone'},
    ]
    for thinking in (True, False):
        kwargs = dict(tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking)
        expected = tokenizer.apply_chat_template(messages, **kwargs)
        assert actual.apply_chat_template(messages, **kwargs) == expected
        assert actual.eos_token_id == tokenizer.eos_token_id
        assert actual.pad_token_id == tokenizer.pad_token_id
        assert len(expected) < 32768
    assert actual.chat_template == (ASSETS / 'chat_template.jinja').read_text(encoding='utf-8')


def test_skyrl_native_generation_header_uses_real_tokenizer(tokenizer):
    path = SOURCES / 'SkyRL-7d94ccf0eac3439c1731ce32018bf043dd639806/skyrl/train/generators/utils.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_generation_prompt_ids')
    namespace = {'Optional': Optional, 'List': List}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    ids = namespace['get_generation_prompt_ids'](tokenizer)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == '<|im_start|>assistant\n<think>\n'


@pytest.mark.parametrize('task', ['textcraft', 'webarena'])
def test_agentgym_real_rollout_handler_model_environment_boundary(tokenizer, task):
    from patch_agentgym_model_entry import patch
    path = SOURCES / 'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL/verl/workers/rollout/schemas.py'
    source = path.read_text(encoding='utf-8')
    original = {}
    fixed = {}
    exec(compile(source, str(path), 'exec'), original)
    patched = patch(source)
    assert patch(patched) == patched
    exec(compile(patched, str(path), 'exec'), fixed)
    # Reproduce the native List[int] contract defect on Transformers 5.
    dummy = SimpleNamespace(messages=[original['Message']('user', 'Read inventory.')])
    assert not isinstance(original['RolloutHandler'].get_generation_prompt(dummy, tokenizer), list)
    messages = [fixed['Message']('user', 'Use the environment.'), fixed['Message']('assistant', 'OK')]
    initial = tokenizer.encode('<|im_start|>user\nUse the environment.<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>', add_special_tokens=False)
    ones, zeros, positions = [1]*len(initial), [0]*len(initial), list(range(len(initial)))
    handler = fixed['RolloutHandler'](messages, task, 0, 0., False,
        initial.copy(), initial.copy(), [], ones.copy(), ones.copy(), [],
        positions.copy(), positions.copy(), [], zeros.copy(), zeros.copy(), [])
    handler.add_user_message(tokenizer, 'inventory: wood')
    assert not any(handler.loss_mask)
    generated_prompt = handler.get_generation_prompt(tokenizer)
    expected = tokenizer.apply_chat_template([m.to_dict() for m in handler.messages],
        add_generation_prompt=True, tokenize=True, return_dict=False)
    assert generated_prompt == expected
    action = 'Thought: inspect.\nAction: inventory'
    before = len(handler.input_ids)
    handler.add_assistant_message(tokenizer, action)
    selected = [x for x, mask in zip(handler.input_ids, handler.loss_mask) if mask]
    assert tokenizer.decode(selected) == action + '<|im_end|>'
    action_end = len(handler.input_ids)
    handler.add_user_message(tokenizer, 'inventory: stone')
    assert len(handler.input_ids) > action_end > before
    assert not any(handler.loss_mask[action_end:])
    assert [m.role for m in handler.messages] == ['user', 'assistant', 'user', 'assistant', 'user']
