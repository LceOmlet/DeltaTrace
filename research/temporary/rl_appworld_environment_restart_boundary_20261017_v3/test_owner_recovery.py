"""Extend the existing native lifecycle tests with the observed restart boundary."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-environment-initialize-retry-20261017-v2/source/test_owner_recovery.py'
code=source.read_text().replace('appworld-environment-initialize-retry-20261017-v2','appworld-environment-restart-boundary-20261017-v3')
code=code.replace('appworld-dtpo-reward-target-exponential-formal-20261017-v1/launch.json',
                  'appworld-dtpo-reward-target-exponential-formal-20261017-v2/launch.json')
code=code.replace("Path(PATCH['patched_file'])","Path(PATCH['owner_root'])/'phi_agents/evals/appworld_evals.py'")
anchor="receipt = dict(status='passed', collected=time.time(), owner_file=owner.__file__,"
assert code.count(anchor)==1
added=r'''
world, episode = Mock(), object()
world.restart.side_effect = AppWorldInitializeTaskError('observed startup readiness failure')
with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=[AppWorldInitializeTaskError('initialize timeout'),episode]) as run:
    assert native(world=world, **inputs) is episode
    assert run.call_count==2 and world.restart.call_count==1
results.append(dict(case='restart_readiness_failure_stays_in_original_budget',passed=True,attempts=2))

world=Mock()
world.restart.side_effect=AppWorldInitializeTaskError('persistent readiness failure')
with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=AppWorldInitializeTaskError('initialize timeout')) as run:
    try:native(world=world, **inputs)
    except ValueError as exc:
        assert 'Exceeded max retries' in str(exc) and run.call_count==budget and world.restart.call_count==budget
    else:raise AssertionError('Original retry exhaustion must remain visible')
results.append(dict(case='restart_failures_do_not_expand_original_budget',passed=True,attempts=budget))

world=Mock()
world.restart.side_effect=RuntimeError('unrelated restart implementation failure')
with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=AppWorldInitializeTaskError('initialize timeout')) as run:
    try:native(world=world, **inputs)
    except RuntimeError as exc:
        assert str(exc)=='unrelated restart implementation failure' and run.call_count==1
    else:raise AssertionError('Unrelated RuntimeError must not be swallowed')
results.append(dict(case='unrelated_restart_RuntimeError_not_swallowed',passed=True))

from phi_agents.appworld import interface
world=interface.AppWorldInterface.__new__(interface.AppWorldInterface)
world._remote_environment_url='http://localhost:39851'
world._max_restarts_on_error=0
world._max_wait_tries=1
world._wait_seconds=0
world._server=Mock()
with patch.object(interface,'logger',Mock()),patch.object(interface.requests,'get',side_effect=requests.ConnectionError('not ready')):
    try:world._wait_for_server_ready()
    except AppWorldInitializeTaskError as exc:
        assert 'Could not connect' in str(exc) and world._server.terminate.call_count==1
    else:raise AssertionError('Native readiness failure must carry its existing initialization exception type')
results.append(dict(case='actual_owner_readiness_failure_is_typed_and_terminates_server',passed=True,environment_starts=0))

world._server=Mock()
response=Mock(status_code=200)
response.json.return_value={'output':{'task_id':interface.DUMMY_TASK_ID}}
with patch.object(interface,'logger',Mock()),patch.object(interface.requests,'get',return_value=response):
    world._wait_for_server_ready()
    assert world._server.terminate.call_count==0
results.append(dict(case='actual_owner_ready_success_unchanged',passed=True,environment_starts=0))
'''
code=code.replace(anchor,added+'\n'+anchor)
exec(compile(code,str(source),'exec'),globals())
