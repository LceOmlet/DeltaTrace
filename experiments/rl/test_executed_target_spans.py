"""CPU-only owner parser parity and exact-ID character-span transport."""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from executed_target_spans import (
    decoded_payload_mask, encoded_payload_mask, full_trajectory_artifact)
from patch_executed_payload_metadata import loop_extractor, textcraft_parser

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / 'research/temporary/rl_upstream_alignment_20260929'
REC = AUDIT / 'recipe-sources'
TEXT = REC / 'AgentGym-d014732d9fe39b975c368c03749bfd50950067f6/agentenv/agentenv/envs/textcraft.py'
LOOP = REC / 'ml-loop/phi_agents/utils/appworld.py'


def owner_function(source, name, namespace, class_name=None):
    tree = ast.parse(source)
    scope = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(AUDIT / 'qwen35-entry-assets', local_files_only=True)


@pytest.mark.parametrize('content', [
    'Thought: inspect.\nAction: inventory',
    'Action:\nget  1 lapis-lazuli!',
    'Thought: 无执行文本',
    'Action: inventory\nAction: get 1 stone',
    'Action: ', 'Action: get 1 🌲 logs',
])
def test_textcraft_original_step_and_native_payload_are_unchanged(content):
    # Load the actual owner schema without importing its unrelated LLM agent.
    spec = importlib.util.spec_from_file_location('native_agentgym_types',
                                                  TEXT.parents[1] / 'controller/types.py')
    types = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = types
    spec.loader.exec_module(types)
    StepOutput = types.StepOutput
    source = TEXT.read_text(encoding='utf-8')
    namespace = dict(re=re, StepOutput=StepOutput)
    original = owner_function(source, 'step', dict(namespace), 'TextCraftEnvClient')
    modified = owner_function(textcraft_parser(source), 'step', dict(namespace), 'TextCraftEnvClient')
    results, posts = [], []
    for step in (original, modified):
        client = SimpleNamespace(_post=Mock(return_value=dict(observation='native', reward=.375, done=True)))
        results.append(step(client, content)); posts.append(client._post.call_args_list)
    assert results[0] == results[1] and posts[0] == posts[1]
    records = []
    client = SimpleNamespace(_post=Mock(return_value=dict(observation='native', reward=.375, done=True)),
                             _executed_payload_sink=records.append)
    assert modified(client, content) == results[0]
    if client._post.call_count:
        assert records[0]['executed_payload'] == client._post.call_args.args[1]['action']
        assert all(content[start:end] for start, end in records[0]['source_spans'])
    else:
        assert records == []


def test_original_textcraft_multiple_actions_reject_before_post_and_target_callback():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('native_agentgym_multiple_action_types',
                                                  TEXT.parents[1] / 'controller/types.py')
    types = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = types
    spec.loader.exec_module(types)
    source = TEXT.read_text(encoding='utf-8')
    content = 'Action: inventory\nAction: get 1 stone'
    outputs = []
    for text in (source, textcraft_parser(source)):
        step = owner_function(text, 'step', dict(re=re, StepOutput=types.StepOutput), 'TextCraftEnvClient')
        post, sink = Mock(), Mock()
        outputs.append(step(SimpleNamespace(_post=post, _executed_payload_sink=sink), content))
        post.assert_not_called()
        sink.assert_not_called()
    assert outputs[0] == outputs[1]
    assert outputs[0].reward == 0 and not outputs[0].done


@pytest.mark.parametrize('content', [
    'Thought: no code',
    '想法👩\n```python\n  apis.supervisor.complete_task()  \n```',
    '```py\na=1\n```\nReason\n```python\nb=2\n```',
    '```python\nx=1\n',
    '```python\n```',
    '```python\nx=1\n```\n```py\n y=2',
])
def test_loop_original_extractor_output_and_source_spans(content):
    source = LOOP.read_text(encoding='utf-8')
    namespace = dict(re=re, logger=logging.getLogger('owner-parser-test'))
    original = owner_function(source, 'extract_code_format_output', dict(namespace))
    modified = owner_function(loop_extractor(source), 'extract_code_format_output', dict(namespace))
    assert modified(content) == original(content)
    code, spans = modified(content, return_source_spans=True)
    assert code == original(content)
    # Include the owner's recognition syntax, as well as its code payload.
    assert all(content[start:end].startswith(('```python', '```py'))
               for start, end in spans if start != end)


def test_native_encode_offsets_leave_original_training_ids_unchanged(tokenizer):
    content = '想法👩: 记录\nAction: get 1 warped stems'
    ids = tokenizer.encode(content, add_special_tokens=False)
    start, end = content.index('get '), len(content)
    selected, metadata = encoded_payload_mask(tokenizer, content, ids, [(start, end)])
    assert ids == tokenizer.encode(content, add_special_tokens=False)
    assert any(selected) and len(selected) == len(ids)
    assert metadata['mapping'] == 'original_native_encoding_offsets'


@pytest.mark.parametrize('noncanonical', [False, True])
@pytest.mark.parametrize('content,span_text', [
    ('想法👩\n```python\napis.supervisor.complete_task()\n```', 'apis.supervisor.complete_task()'),
    ('prefix中👩文suffix', '中👩文'),
    (' abcdef ', 'bcd'),
])
def test_original_decoder_boundaries_cover_utf8_and_noncanonical_ids(tokenizer, noncanonical, content, span_text):
    ids = (sum((tokenizer.encode(char, add_special_tokens=False) for char in content), [])
           if noncanonical else tokenizer.encode(content, add_special_tokens=False))
    assert tokenizer.decode(ids) == content
    original = ids.copy()
    start, end = content.index(span_text), content.index(span_text) + len(span_text)
    mask, metadata = decoded_payload_mask(tokenizer, content, ids, [(start, end)])
    assert ids == original
    first, last = metadata['ranges'][0]['token_span']
    before = tokenizer.decode(ids[:first])
    through = tokenizer.decode(ids[:last])
    assert content[:start].startswith(before)
    assert through.startswith(content[:end])
    assert any(mask) and mask == [first <= i < last for i in range(len(ids))]


def test_full_artifact_keeps_truncated_execution_and_excludes_synthetic_and_padding():
    full_ids = [10, 11, 20, 21, 30, 31, 40]
    artifact = full_trajectory_artifact([10, 11], full_ids,
        [0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1, 0], [20, 21, 99, 0])
    assert artifact['response_ids'] == [20, 21, 30, 31, 40]
    assert artifact['retained_response_positions'] == [0, 1, -1, -1]
    assert artifact['target_mask'] == [0, 1, 0, 1, 0]
    app = full_trajectory_artifact([10, 11], full_ids,
        [0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1, 0], full_ids[1:4], retained_start=1)
    assert app['retained_response_positions'] == [-1, 0, 1]


def test_actual_textcraft_recording_handler_preserves_history_before_owner_truncation(tokenizer):
    """Call the original handler and the bridge's actual nested subclass."""
    from typing import List, Literal
    source = (REC / 'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/'
              'AgentGym-RL/verl/workers/rollout/schemas.py').read_text(encoding='utf-8')
    owner_tree = ast.parse(source)
    classes = [node for node in owner_tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(List=List, Literal=Literal, PreTrainedTokenizer=object)
    exec(compile(ast.Module(body=classes, type_ignores=[]), 'official-RolloutHandler', 'exec'), namespace)
    original = namespace['RolloutHandler']
    bridge_tree = ast.parse((REPO / 'experiments/rl/textcraft_owner_rollout.py').read_text(encoding='utf-8'))
    recorder = next(node for node in ast.walk(bridge_tree)
                    if isinstance(node, ast.ClassDef) and node.name == 'RecordedHandler')
    handlers, records = [], [[{}]]
    scope = dict(namespace, original_handler=original, handlers=handlers, records=records)
    exec(compile(ast.Module(body=[recorder], type_ignores=[]), 'actual-RecordedHandler', 'exec'), scope)
    prefix = tokenizer.encode('\n<|im_start|>assistant\n', add_special_tokens=False)
    options = dict(messages=[], task_name='textcraft', item_id=31, score=0, done=False,
        input_ids=prefix.copy(), prompt_ids=prefix.copy(), response_ids=[],
        attention_mask=[1] * len(prefix), prompt_attention_mask=[1] * len(prefix),
        response_attention_mask=[], position_ids=list(range(len(prefix))),
        prompt_position_ids=list(range(len(prefix))), response_position_ids=[],
        loss_mask=[0] * len(prefix), prompt_loss_mask=[0] * len(prefix), response_loss_mask=[],
        max_response_len=6, max_model_len=len(prefix) + 6)
    from copy import deepcopy
    plain, recorded = original(**deepcopy(options)), scope['RecordedHandler'](**deepcopy(options))
    content = 'Thought: inspect.\nAction: get 1 warped stems'
    for handler in (plain, recorded):
        handler.add_assistant_message(tokenizer, content)
        handler.add_user_message(tokenizer, 'Got 1 warped stems')
    full_ids, full_mask = plain.input_ids.copy(), plain.loss_mask.copy()
    plain.truncate_output_ids(); recorded.truncate_output_ids()
    assert recorded.dt_full_input_ids == full_ids and recorded.dt_full_policy_mask == full_mask
    for key in ('input_ids', 'response_ids', 'attention_mask', 'position_ids', 'loss_mask', 'response_loss_mask'):
        assert getattr(plain, key) == getattr(recorded, key)
    assert len(recorded.dt_full_input_ids) > len(recorded.input_ids)
    assert records[0][0]['native_content'] == content


def test_patches_are_idempotent_and_do_not_add_another_parser():
    for path, patch in ((TEXT, textcraft_parser), (LOOP, loop_extractor)):
        source = path.read_text(encoding='utf-8')
        assert patch(patch(source)) == patch(source)
        old_tree, new_tree = ast.parse(source), ast.parse(patch(source))
        old = {(type(node).__name__, getattr(node, 'name', None)) for node in old_tree.body}
        new = {(type(node).__name__, getattr(node, 'name', None)) for node in new_tree.body}
        assert old == new  # Extend the original owner; no second parser entry.
