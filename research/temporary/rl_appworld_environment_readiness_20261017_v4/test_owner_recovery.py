"""Reuse native lifecycle checks and cover the demonstrated delayed readiness."""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/appworld-environment-restart-boundary-20261017-v3/source/test_owner_recovery.py'
code = source.read_text().replace('appworld-environment-restart-boundary-20261017-v3',
                                'appworld-environment-readiness-20261017-v4')
# The reused test's launch environment is v2. Apply the environment substitution
# after that original test has been loaded by the reused v3 wrapper.
tail = "exec(compile(code,str(source),'exec'),globals())"
assert code.count(tail) == 1
code = code.replace(tail,
    '''code=code.replace("PATCH['baseline']", "str(ROOT/'candidates/appworld-environment-initialize-retry-20261017-v2/loop-owner')")
''' + tail)
anchor = "code=code.replace(anchor,added+'\\n'+anchor)"
assert code.count(anchor) == 1
extra = r'''
response=Mock(status_code=200)
response.json.return_value={'output':{'task_id':interface.DUMMY_TASK_ID}}
process=Mock()
with patch.object(interface,'get_free_port',return_value=39851), \
     patch.object(interface,'launch_appworld_environment_server',return_value=process) as launch, \
     patch.object(interface.time,'sleep'), \
     patch.object(interface.requests,'get',side_effect=[requests.ConnectionError('still importing')]*6+[response]) as get:
    world=interface.AppWorldInterface(stdout_to_devnull=True)
    assert world._max_wait_tries==100 and world._wait_seconds==1.0
    assert get.call_count==7 and launch.call_count==1 and process.terminate.call_count==0
    assert world._max_restarts_on_error==2 and world.timeout_seconds==100
results.append(dict(case='actual_constructor_delayed_readiness_uses_existing_timeout_without_restart',passed=True,polls=7,environment_starts=0))

with patch.object(interface,'get_free_port',return_value=39851), \
     patch.object(interface,'launch_appworld_environment_server',return_value=Mock()), \
     patch.object(interface.time,'sleep'), \
     patch.object(interface.requests,'get',return_value=response) as get:
    world=interface.AppWorldInterface(stdout_to_devnull=True)
    assert get.call_count==1
results.append(dict(case='actual_constructor_immediate_success_unchanged',passed=True,environment_starts=0))
'''
code = code.replace(anchor, "added += " + repr(extra) + "\n" + anchor)
exec(compile(code, str(source), 'exec'), globals())
