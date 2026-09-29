"""Real official SQLEnv, SQLite, and model tokenizer; no generation fixture score."""
from copy import deepcopy
import os
from pathlib import Path
import sqlite3

import pytest
from transformers import AutoTokenizer
from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig
from owner_environment_transport import PolicyReply
from sql_environment_entry import SQLSession


@pytest.fixture(scope='module')
def tokenizer():
    assets = os.environ.get('DT_TOKENIZER_PATH', str(Path(__file__).resolve().parents[2] /
        'research/temporary/rl_upstream_alignment_20260929/qwen35-entry-assets'))
    return AutoTokenizer.from_pretrained(assets, local_files_only=True)


@pytest.fixture
def task(tmp_path):
    db = tmp_path / 'spider/database/fixture/fixture.sqlite'
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.execute('create table items (name text)')
        con.execute("insert into items values ('wood')")
    return dict(prompt=[dict(role='user', content='Return the names from items(name).')],
                extras=dict(db_id='fixture', data='spider', reward_spec={'ground_truth': 'select name from items'}))


@pytest.mark.parametrize('action,expected', [
    ('<think>Read names.</think><solution>select name from items</solution>', 1.),
    ('<think>Wrong.</think><solution>select 1</solution>', 0.),
    ('<solution>select name from items</solution>', -1.),
])
def test_original_reward_and_termination_are_passed_through(tokenizer, tmp_path, task, action, expected):
    session = SQLSession.create(tokenizer, str(tmp_path), task, 6, 29000)
    direct = SQLEnv(Text2SQLEnvConfig(str(tmp_path)), dict(task['extras'], max_turns=6))
    direct.init(deepcopy(task['prompt']))
    ids = tokenizer.encode(action, add_special_tokens=False)
    result = session.step(PolicyReply(ids, [-.2]*len(ids), action, 'stop'))
    assert result == direct.step(action)
    assert result['reward'] == expected and session.done
    assert session.env.chat_history == direct.chat_history


def test_native_tool_observation_and_six_step_horizon(tokenizer, tmp_path, task):
    session = SQLSession.create(tokenizer, str(tmp_path), task, 6, 29000)
    initial = session.prompt_ids.copy()
    text = '<think>Inspect.</think><sql>select name from items</sql>'
    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    result = session.step(PolicyReply(ids, [-.1]*len(ids), text, 'stop'))
    obs_ids = [t for msg in result['observations'] for t in tokenizer.encode(msg['content'], add_special_tokens=False)]
    assert 'wood' in str(result['observations'])
    assert session.prompt_ids == initial + ids[:-1] + obs_ids
    assert result['reward'] == 0 and not session.done
    for i in range(5):
        result = session.step(PolicyReply(ids, [-.1]*len(ids), text, 'stop'))
        assert session.done == (i == 4)
    assert session.env.turns == 6


def test_context_limit_ends_without_extra_action_or_synthetic_reward(tokenizer, tmp_path, task):
    session = SQLSession.create(tokenizer, str(tmp_path), task, 6, 29000)
    session.max_input_length = len(session.prompt_ids)
    text = '<think>Inspect.</think><sql>select name from items</sql>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = session.step(PolicyReply(ids, [-.1]*len(ids), text, 'stop'))
    assert session.done and session.env.turns == 1
    assert result['reward'] == 0 and not result['done']
