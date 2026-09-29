"""Native LOOP environment behavior; model and HTTP world are explicit fixtures.

No trainer, DT, reward predictor, AppWorld server or model is instantiated.
The actual owner constructs messages, parses actions and runs the episode loop.
"""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from threading import Lock

import cattrs
import pytest

from loop_owner_recipe import environment_configuration
from phi_agents.appworld.interface import (
    Task, Supervisor, AppWorldTaskEvalResult, Pass, Failure, load_task_ids,
)
from phi_agents.evals.appworld_evals import _run_vllm_inference_single_server_single_task
from phi_agents.inference.config import AppWorldConfig
from phi_agents.rl.type_defs import PolicyMessage
from phi_agents.rl.vllm_client import MaxSeqLenExceeded
from phi_agents.utils.appworld import extract_code_format_output

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / 'research/temporary/rl_upstream_alignment_20260929'
OWNER = RECEIPTS / 'recipe-sources/ml-loop'


def test_native_environment_configuration_matches_saved_author_config():
    expected = json.loads((RECEIPTS / 'loop-environment-config.json').read_text())['environment']
    assert environment_configuration(OWNER) == expected


def test_native_training_and_evaluation_task_splits():
    train = load_task_ids('train_difficulty_1_and_train_difficulty_2')
    evaluation = load_task_ids('dev')
    assert len(train) == len(set(train)) == 72
    assert len(evaluation) == len(set(evaluation)) == 57
    assert set(train).isdisjoint(evaluation)


@pytest.mark.parametrize('mode,success,expected', [
    ('training', False, .5), ('evaluation', False, 0.),
    ('training', True, 1.), ('evaluation', True, 1.),
])
def test_native_runner_training_fraction_and_evaluation_success(monkeypatch, mode, success, expected):
    from phi_agents.rl import appworld_scenario_runner as owner
    from phi_agents.evals.appworld_evals import Episode, TaskEvalResult
    from phi_agents.rl.type_defs import PolicyTokenInfo, UserMessage
    environment = environment_configuration(OWNER)
    config = cattrs.structure(environment[f'{mode}_environment']['appworld_config'], AppWorldConfig)
    task = Task('fixture_1', datetime(2026, 1, 2), 'Read inventory.',
                Supervisor('Test', 'User', '000', 'fixture@example.invalid'))
    result = AppWorldTaskEvalResult(success, 2, 2,
        [Pass('passed', None)] * (2 if success else 1), [] if success else [Failure('failed', '', None)])
    episode = Episode('fixture', task, [UserMessage('inventory')],
        TaskEvalResult.create(result, num_interactions=3), 1, 2, 2, False)
    run_episode = Mock(return_value=episode)
    monkeypatch.setattr(owner, 'run_vllm_inference_single_server_single_task', run_episode)
    runner = owner.AppWorldScenarioRunner.__new__(owner.AppWorldScenarioRunner)
    runner.appworld_config = config
    runner.lock = Lock()
    runner.world = SimpleNamespace(server=object(), port=1, remote_environment_url='fixture://world')
    llm = SimpleNamespace(get_policy_token_info=Mock(return_value=PolicyTokenInfo()))
    dataset = environment[f'{mode}_task_sampler']['dataset_name']
    rollout = runner.run(owner.AppWorldScenario(task_id=task.task_id, dataset_name=dataset), llm)
    assert rollout.ret == expected
    assert rollout.appworld_rollout_data.eval_result is episode.eval_result
    assert rollout.messages is episode.chat_history
    assert run_episode.call_args.kwargs['appworld_config'] is config
    assert config.env.no_code_found_penalty == config.env.execution_failed_penalty == 0


def test_native_sampler_consumes_each_task_once_before_reshuffle():
    from phi_agents.rl.appworld_scenario_runner import AppWorldScenarioSampler
    for dataset in ('train_difficulty_1_and_train_difficulty_2', 'dev'):
        ids = load_task_ids(dataset)
        sampler = AppWorldScenarioSampler(dataset, seed=17)
        for _ in range(2):
            samples = [next(sampler) for _ in ids]
            assert {s.task_id for s in samples} == set(ids)
            assert all(s.dataset_name == dataset for s in samples)


@pytest.mark.parametrize('text,expected', [
    ('```python\nprint(1)\n```', 'print(1)\n'),
    ('```py\nprint(1)\n```\n```python\nprint(2)\n```', 'print(1)\nprint(2)\n'),
    ('<code>print(1)</code>', ''),
    ('No code here', ''),
])
def test_original_action_parser(text, expected):
    assert extract_code_format_output(text) == expected


class GenerationFixture:
    """Finite replies only; token IDs are fixtures, not a real tokenizer test."""
    special_tokens = {}

    def __init__(self, budget_at=None):
        self.budget_at = budget_at
        self.inputs = []

    def get_tokens(self, messages):
        return [1], None, None

    def generate(self, messages):
        self.inputs.append([(m.role, m.content) for m in messages])
        if len(self.inputs) == self.budget_at:
            raise MaxSeqLenExceeded('explicit generation-boundary fixture')
        return PolicyMessage('```python\nprint(1)\n```', prompt_tokens=[],
            generated_tokens=[101], generated_token_logprobs=[-.1],
            stopped_by_max_tokens_limit=False)


@pytest.mark.parametrize('mode,stop', [
    ('training', 'horizon'), ('evaluation', 'horizon'),
    ('training', 'complete'), ('training', 'budget'),
])
def test_original_agent_history_and_episode_termination(mode, stop):
    native = environment_configuration(OWNER)[f'{mode}_environment']['appworld_config']
    config = cattrs.structure(native, AppWorldConfig)
    task = Task('fixture_1', datetime(2026, 1, 2), 'Read the inventory.',
                Supervisor('Test', 'User', '000', 'fixture@example.invalid'))
    evaluation = AppWorldTaskEvalResult(False, 2, 2,
        [Pass('fixture passed', None)], [Failure('fixture failed', '', None)])
    world = SimpleNamespace(
        initialize=Mock(return_value=task), execute=Mock(return_value='inventory result'),
        task_completed=Mock(side_effect=[False, False, True] if stop == 'complete' else None,
                            return_value=False),
        evaluate=Mock(return_value=evaluation), close_world=Mock())
    llm = GenerationFixture(budget_at=3 if stop == 'budget' else None)
    episode = _run_vllm_inference_single_server_single_task(
        world, task.task_id, 'local-unit-fixture', config, llm, True)
    executed = {'horizon': 40, 'complete': 3, 'budget': 2}[stop]
    assert world.execute.call_count == executed
    assert world.initialize.call_args.kwargs['raise_on_unsafe_syntax'] is False
    assert all(c.args == ('print(1)\n',) for c in world.execute.call_args_list)
    world.evaluate.assert_called_once_with()
    world.close_world.assert_called_once_with()
    assert episode.eval_result.num_tests == 2 and len(episode.eval_result.passes) == 1
    assert not episode.cancelled
    assert len(llm.inputs[0]) == 24  # Original few-shot prompt, not one flattened user string.
    assert all(len(messages) == 24 + 2 * i for i, messages in enumerate(llm.inputs))
    for messages in llm.inputs[1:]:
        assert messages[:24] == llm.inputs[0]
        assert 'inventory result' in messages[-1][1]
    if stop == 'budget':
        assert episode.chat_history[-1].content == 'Terminating episode: exceeded max sequence length'
