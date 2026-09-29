"""Use the original readiness/restart loop for the observed empty HTTP 404."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import requests
from phi_agents.appworld import interface


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
