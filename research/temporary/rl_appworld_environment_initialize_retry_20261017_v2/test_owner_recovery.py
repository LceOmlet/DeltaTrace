"""Exercise the actual LOOP recovery function; no model or environment is started."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'appworld-environment-initialize-retry-20261017-v2'
AUDIT = ROOT / 'receipts' / VERSION
PATCH = json.loads((AUDIT / 'owner-patch.json').read_bytes())
if '--native' not in sys.argv:
    original = json.loads((ROOT / 'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v1/launch.json').read_bytes())
    env = dict(os.environ)
    env.update(original['effective_environment'])
    env = {key: value.replace(PATCH['baseline'], PATCH['owner_root']) for key, value in env.items()}
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--native'], env=env))

import site
if os.environ.get('LOOP_EXTRAS'):
    site.addsitedir(os.environ['LOOP_EXTRAS'])
import requests
from unittest.mock import Mock, patch
from phi_agents.evals import appworld_evals as owner
from phi_agents.appworld.interface import AppWorldExecutionError, AppWorldInitializeTaskError

assert Path(owner.__file__) == Path(PATCH['patched_file'])
owner.logger = Mock()
native = owner.run_vllm_inference_single_server_single_task
budget = inspect.signature(native).parameters['max_retries'].default
results = []
inputs = dict(task_id='60d0b5b_1', experiment_name='owner-recovery-contract',
              appworld_config=object(), llm=object(), with_evaluation=True)

for name, failure in [('initialize_ReadTimeout', AppWorldInitializeTaskError('same observed initialization failure')),
                      ('raw_ReadTimeout', requests.ReadTimeout('same observed transport failure')),
                      ('existing_execution_error', AppWorldExecutionError()),
                      ('existing_HTTP_error', requests.HTTPError()),
                      ('existing_connection_error', requests.ConnectionError())]:
    world, episode = Mock(), object()
    with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=[failure, episode]) as run:
        result = native(world=world, **inputs)
        assert result is episode and run.call_count == 2 and world.restart.call_count == 1
        assert all(call.kwargs == dict(world=world, **inputs) for call in run.call_args_list)
    results.append(dict(case=name, passed=True, attempts=2, restarts=1))

world, episode = Mock(), object()
with patch.object(owner, '_run_vllm_inference_single_server_single_task', return_value=episode) as run:
    assert native(world=world, **inputs) is episode
    assert run.call_count == 1 and world.restart.call_count == 0
results.append(dict(case='native_success_unchanged', passed=True))

world = Mock()
with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=RuntimeError('unrelated')) as run:
    try:
        native(world=world, **inputs)
    except RuntimeError as exc:
        assert str(exc) == 'unrelated' and run.call_count == 1 and world.restart.call_count == 0
    else:
        raise AssertionError('Unrelated error must remain visible')
results.append(dict(case='unrelated_error_not_swallowed', passed=True))

world = Mock()
with patch.object(owner, '_run_vllm_inference_single_server_single_task', side_effect=AppWorldInitializeTaskError('persistent')) as run:
    try:
        native(world=world, **inputs)
    except ValueError as exc:
        assert 'Exceeded max retries' in str(exc) and run.call_count == budget and world.restart.call_count == budget
    else:
        raise AssertionError('Original finite exhaustion behavior must remain visible')
results.append(dict(case='original_retry_budget_and_exhaustion', passed=True, attempts=budget))
receipt = dict(status='passed', collected=time.time(), owner_file=owner.__file__,
               cases=results, original_retry_budget=budget, model_calls=0, environment_starts=0,
               optimizer_updates=0, scope='Native recovery lifecycle only; no training quality or numerical DT claim')
(AUDIT / 'CPU-recovery-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
