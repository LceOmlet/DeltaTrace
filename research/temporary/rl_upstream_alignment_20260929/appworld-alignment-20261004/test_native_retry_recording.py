"""CPU diagnostics: unchanged pinned LOOP retry and bridge recording AST.

Only the world and one-episode I/O are fixtures. No retry, task service,
model, training algorithm, tokenizer, or reward implementation is copied.
"""
import ast
from pathlib import Path
import subprocess
from threading import local
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


REPO = Path(__file__).resolve().parents[4]
OWNER = REPO / 'research/temporary/rl_upstream_alignment_20260929/recipe-sources/ml-loop'
BRIDGE = REPO / 'experiments/rl/loop_owner_worker.py'
BASELINE_COMMIT = 'ea2d32a3359c29100704dde1c73efb4641b14669'


def recorded_class(source):
    tree = ast.parse(source)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'recorded_runner')
    return next(n for n in factory.body if isinstance(n, ast.ClassDef))


@pytest.fixture(scope='module')
def sources():
    original = subprocess.run(['git', 'show', BASELINE_COMMIT + ':experiments/rl/loop_owner_worker.py'],
                              cwd=REPO, check=True, capture_output=True, text=True).stdout
    current = BRIDGE.read_text(encoding='utf-8')
    return original, current


def run_case(source, failed_attempts=0, restart_error=None):
    owner_path = OWNER / 'phi_agents/evals/appworld_evals.py'
    retry_node = next(n for n in ast.parse(owner_path.read_text(encoding='utf-8')).body
                     if isinstance(n, ast.FunctionDef)
                     and n.name == 'run_vllm_inference_single_server_single_task')
    recording = local()
    class AppWorldExecutionError(Exception):
        pass
    class HTTPError(Exception):
        pass
    class ConnectionError(Exception):
        pass
    marker = object()
    original_restart = Mock(return_value=marker, side_effect=restart_error)
    world = SimpleNamespace(execute=Mock(), restart=original_restart)
    attempts = []
    def one_episode(**kwargs):
        index = len(attempts)
        prompt, ids = [10, 20, 30], [101+index, 248046]
        attempts.append((prompt, ids))
        recording.requests.append((f'attempt-{index}', len(prompt), len(ids)))
        world.execute()
        if index < failed_attempts:
            raise AppWorldExecutionError('fixture crash after a completed response')
        return SimpleNamespace(chat_history=SimpleNamespace(tokens=prompt+ids),
                               completion_requests=None, task='fixture')
    namespace = dict(AppWorldExecutionError=AppWorldExecutionError,
        requests=SimpleNamespace(HTTPError=HTTPError, ConnectionError=ConnectionError),
        _run_vllm_inference_single_server_single_task=one_episode, logger=Mock())
    retry_module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), retry_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(retry_module), str(owner_path), 'exec'), namespace)
    retry = namespace['run_vllm_inference_single_server_single_task']
    class RunnerFixture:
        def __init__(self, **options):
            self.world = world
        def run(self, scenario, llm):
            return retry(world=world, task_id='fixture', experiment_name='diagnostic',
                         appworld_config=None, llm=None, with_evaluation=True)
    namespace = dict(AppWorldScenarioRunner=RunnerFixture, _recording=recording)
    exec(compile(ast.Module(body=[recorded_class(source)], type_ignores=[]), str(BRIDGE), 'exec'), namespace)
    runner = namespace['RecordedRunner']()
    case = SimpleNamespace(runner=runner, world=world, original_restart=original_restart,
                           marker=marker, attempts=attempts, recording=recording)
    if restart_error is not None:
        with pytest.raises(type(restart_error), match=str(restart_error)):
            runner.run(None, None)
    else:
        case.result = runner.run(None, None)
    return case


def assert_source_token_alignment(case):
    # Same exact token-list equality as to_batch's tensor assertion, without torch.
    responses = {f'attempt-{i}': ids for i, (_, ids) in enumerate(case.attempts)}
    for key, start, length in case.result.completion_requests:
        assert case.result.chat_history.tokens[start:start+length] == responses[key][:length]


def test_committed_baseline_reproduces_discarded_episode_mismatch(sources):
    case = run_case(sources[0], failed_attempts=1)
    assert case.original_restart.call_count == 1
    assert case.result.completion_requests == [('attempt-0', 3, 2), ('attempt-1', 3, 2)]
    assert case.result.chat_history.tokens[3:] == case.attempts[-1][1]
    with pytest.raises(AssertionError):
        assert_source_token_alignment(case)


def test_no_retry_retains_identical_owner_artifacts(sources):
    baseline, candidate = [run_case(source) for source in sources]
    for case in (baseline, candidate):
        assert case.original_restart.call_count == 0
        assert_source_token_alignment(case)
    assert baseline.result.completion_requests == candidate.result.completion_requests
    assert baseline.result.chat_history.tokens == candidate.result.chat_history.tokens
    assert baseline.result.tool_execution_count == candidate.result.tool_execution_count == 1
    old_methods = {n.name: ast.dump(n) for n in recorded_class(sources[0]).body if isinstance(n, ast.FunctionDef)}
    new_methods = {n.name: ast.dump(n) for n in recorded_class(sources[1]).body if isinstance(n, ast.FunctionDef)}
    assert old_methods['run'] == new_methods['run']


@pytest.mark.parametrize('failed_attempts', [1, 2, 4])
def test_native_successful_retries_keep_only_retained_episode(sources, failed_attempts):
    case = run_case(sources[1], failed_attempts)
    assert len(case.attempts) == failed_attempts + 1
    assert case.original_restart.call_count == failed_attempts
    assert case.result.completion_requests == [(f'attempt-{failed_attempts}', 3, 2)]
    assert case.result.tool_execution_count == 1
    assert_source_token_alignment(case)


def test_original_restart_failure_does_not_erase_recorded_attempt(sources):
    case = run_case(sources[1], failed_attempts=1,
                    restart_error=RuntimeError('fixture restart failed'))
    assert len(case.attempts) == 1
    assert case.original_restart.call_count == 1
    assert case.recording.requests == [('attempt-0', 3, 2)]
    assert case.recording.executions == 1


def test_original_restart_arguments_and_return_value_are_preserved(sources):
    case = run_case(sources[1])
    assert case.world.restart('fixture', reason='retry') is case.marker
    case.original_restart.assert_called_once_with('fixture', reason='retry')
    assert case.recording.requests == [] and case.recording.executions == 0