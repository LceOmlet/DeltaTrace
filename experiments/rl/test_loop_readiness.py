"""Call LOOP's actual owner method; only external HTTP/time/server I/O is mocked.

Run in the existing phi_agents CPU environment. Source-only execution on a
machine without phi_agents is separate evidence, not installed-owner pytest.
"""
import ast
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import requests
from phi_agents.appworld import interface

from patch_loop_readiness import patch


def response(status,body):
    result=requests.Response()
    result.status_code=status
    result._content=body
    return result


def test_non_appworld_http_reply_uses_owner_restart(monkeypatch):
    world=object.__new__(interface.AppWorldInterface)
    world._remote_environment_url='http://localhost:30497'
    world._server=SimpleNamespace(terminate=Mock())
    world._max_restarts_on_error=1
    world._max_wait_tries=2
    world._wait_seconds=0
    world._init_server=Mock()
    get=Mock(side_effect=[response(404,b''),response(404,b''),
        response(200,json.dumps(dict(output=dict(task_id=interface.DUMMY_TASK_ID))).encode())])
    monkeypatch.setattr(interface.requests,'get',get)
    world._wait_for_server_ready()
    assert get.call_count==3
    world._init_server.assert_called_once_with()
    world.server.terminate.assert_called_once_with()


def test_ready_owner_response_does_not_restart(monkeypatch):
    world=object.__new__(interface.AppWorldInterface)
    world._remote_environment_url='http://localhost:15000'
    world._max_restarts_on_error=2
    world._max_wait_tries=5
    world._init_server=Mock()
    monkeypatch.setattr(interface.requests,'get',lambda *args: response(200,
        json.dumps(dict(output=dict(task_id=interface.DUMMY_TASK_ID))).encode()))
    world._wait_for_server_ready()
    world._init_server.assert_not_called()


def owner_world(restarts=0):
    world=object.__new__(interface.AppWorldInterface)
    world._remote_environment_url='http://localhost:15000'
    world._server=SimpleNamespace(terminate=Mock())
    world._max_restarts_on_error=restarts
    world._max_wait_tries=5
    world._wait_seconds=1.0
    world._init_server=Mock()
    return world


class IOClock:
    """An external ready time; it does not implement readiness/retry behavior."""
    def __init__(self, ready_at):
        self.ready_at=ready_at
        self.now=0.0
        self.get_times=[]
        self.waits=[]

    def get(self, *args, **kwargs):
        self.get_times.append(self.now)
        if self.ready_at is None or self.now < self.ready_at:
            raise requests.exceptions.ConnectionError('Original server not ready yet')
        return response(200,json.dumps(dict(output=dict(task_id=interface.DUMMY_TASK_ID))).encode())

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now+=seconds


def attach_clock(monkeypatch, clock):
    monkeypatch.setattr(interface.requests,'get',clock.get)
    monkeypatch.setattr(interface.time,'sleep',clock.sleep)


@pytest.mark.parametrize('ready_at,expected_times,waits',[
    (0.0,[0.0],[]), (2.0,[0.0,1.0,2.0],[1.0,1.0])])
def test_early_ready_keeps_original_poll_and_wait(monkeypatch,ready_at,expected_times,waits):
    world=owner_world()
    clock=IOClock(ready_at)
    attach_clock(monkeypatch,clock)
    world._wait_for_server_ready()
    assert clock.get_times==expected_times
    assert clock.waits==waits
    world._init_server.assert_not_called()
    world.server.terminate.assert_not_called()


def test_never_ready_keeps_five_checks_and_five_waits_per_restart(monkeypatch):
    world=owner_world(restarts=2)
    clock=IOClock(None)
    attach_clock(monkeypatch,clock)
    with pytest.raises(RuntimeError,match='Could not connect'):
        world._wait_for_server_ready()
    assert clock.get_times==[0.,1.,2.,3.,5.,5.,6.,7.,8.,10.,10.,11.,12.,13.,15.]
    assert clock.waits==[1.0]*15
    assert clock.now==15.0
    assert world._init_server.call_count==2
    assert world.server.terminate.call_count==3


def test_ready_during_last_original_wait_is_checked(monkeypatch):
    world=owner_world()
    clock=IOClock(4.5)
    attach_clock(monkeypatch,clock)
    world._wait_for_server_ready()
    assert clock.get_times==[0.,1.,2.,3.,5.]
    assert clock.waits==[1.0]*5
    assert clock.now==5.0
    world._init_server.assert_not_called()
    world.server.terminate.assert_not_called()


def test_same_clock_reproduces_old_failure_and_fixed_success(monkeypatch):
    # Extract the actual installed owner body, undo only this two-site patch,
    # and execute the original function. No readiness loop is copied here.
    source=inspect.getsource(interface)
    original=source.replace(
        '                    if attempt_wait == self._max_wait_tries - 1:\n'
        '                        time.sleep(self._wait_seconds)\n', '', 1).replace(
        '                    if attempt_wait < self._max_wait_tries - 1:\n'
        '                        time.sleep(self._wait_seconds)\n',
        '                    time.sleep(self._wait_seconds)\n', 1)
    assert patch(original)==source
    assert patch(source)==source
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef)
              and n.name=='AppWorldInterface')
    method=next(n for n in node.body if isinstance(n,ast.FunctionDef)
                and n.name=='_wait_for_server_ready')
    namespace=dict(vars(interface))
    exec(compile(ast.Module(body=[method],type_ignores=[]),interface.__file__,'exec'),namespace)
    old_clock=IOClock(4.5)
    attach_clock(monkeypatch,old_clock)
    with pytest.raises(RuntimeError,match='Could not connect'):
        namespace['_wait_for_server_ready'](owner_world())
    new_clock=IOClock(4.5)
    attach_clock(monkeypatch,new_clock)
    owner_world()._wait_for_server_ready()
    assert old_clock.get_times==[0.,1.,2.,3.,4.]
    assert new_clock.get_times==[0.,1.,2.,3.,5.]
    assert old_clock.waits==new_clock.waits==[1.0]*5
    assert old_clock.now==new_clock.now==5.0
