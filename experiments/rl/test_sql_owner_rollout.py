"""Real SQLEnv/SQLite parity against the unmodified SkyRL agent_loop."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

from owner_environment_transport import PolicyReply
from sql_environment_entry import configure_sql_tokenizer
from sql_owner_rollout import SQLRollouts, Request, Result


@pytest.fixture(scope='module')
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.environ['DT_TOKENIZER_PATH'], local_files_only=True)
    configure_sql_tokenizer(tokenizer, Path(__file__).with_name('qwen3_acc_thinking.jinja2'))
    return tokenizer


@pytest.fixture
def task(tmp_path):
    db = tmp_path / 'spider/database/fixture/fixture.sqlite'
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.execute('create table items (name text)')
        con.execute("insert into items values ('wood')")
    return dict(prompt=[dict(role='user', content='Return names from items(name).')],
                extras=dict(db_id='fixture', data='spider', reward_spec={'ground_truth':'select name from items'}))


class NativeFixtureClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, batch, model=None):
        self.requests.append(deepcopy(batch))
        reply = next(self.replies)
        return dict(responses=[reply.text], response_ids=[reply.token_ids.copy()],
                    response_logprobs=[reply.logprobs.copy()], stop_reasons=[reply.finish_reason])

    async def finish_session(self, session_id):
        pass


@pytest.mark.parametrize('evaluation', [False, True])
@pytest.mark.parametrize('case', ['success', 'invalid', 'six_turns', 'length'])
def test_native_complete_trajectory_is_unchanged(tokenizer, task, tmp_path, evaluation, case):
    native = json.loads(Path(__file__).with_name('owner_environment_configs.json').read_text())['SkyRL-SQL']
    sql = '<think>Inspect.</think><sql>select name from items</sql>'
    final = '<think>Answer.</think><solution>select name from items</solution>'
    texts = [sql, final]
    if case == 'invalid':
        texts = ['<solution>select name from items</solution>']
    elif case == 'six_turns':
        texts = [sql] * 6
    elif case == 'length':
        native['generator']['max_input_length'] = len(tokenizer.apply_chat_template(
            task['prompt'], add_generation_prompt=True, tokenize=True, return_dict=False))
        texts = [sql]
    replies = []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Include a sampled EOS on alternating turns; stop-string responses do
        # not necessarily contain one. The owner decides which EOS to retain.
        if i % 2 == 0:
            ids.append(tokenizer.eos_token_id)
        replies.append(PolicyReply(ids, [-.2] * len(ids), text, 'stop'))

    fixture = NativeFixtureClient(replies)
    cfg = GeneratorConfig.from_dict_config(OmegaConf.create(native['generator']))
    gym = SkyRLGymConfig()
    gym.text2sql.db_path = str(tmp_path)
    original = SkyRLGymGenerator(cfg, gym, fixture, tokenizer)
    expected = asyncio.run(original.agent_loop(
        prompt=deepcopy(task['prompt']), env_class='text2sql', env_extras=deepcopy(task['extras']),
        max_tokens=3000, max_input_length=cfg.max_input_length,
        sampling_params=native['eval_sampling' if evaluation else 'sampling'],
        training_phase='eval' if evaluation else 'train'))
    original.env_executor.shutdown(wait=True)
    # SQLEnv scores the complete history on its sixth turn. Missing <solution>
    # is the owner's -1 format reward; input-budget termination remains zero.
    assert sum(expected.reward) == {'success':1., 'invalid':-1., 'six_turns':-1., 'length':0.}[case]

    bridge = SQLRollouts(tokenizer, native, tmp_path, evaluation=evaluation)
    try:
        event = bridge.start([deepcopy(task)])[0]
        for i, reply in enumerate(replies):
            assert isinstance(event, Request)
            assert event.prompt_ids == fixture.requests[i]['prompt_token_ids'][0]
            bridge.reply(event, reply)
            event = bridge.next_event(0)
        assert isinstance(event, Result)
        got, want = asdict(event.trajectory), asdict(expected)
        for key in ['e2e_time', 'time_splits']:
            got.pop(key)
            want.pop(key)
        assert got == want
        assert len(bridge.records[0]) == len(replies)
        assert sum(r['reward'] for r in bridge.records[0]) == sum(expected.reward)
    finally:
        bridge.close()
