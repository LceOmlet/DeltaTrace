"""Compare the callback's request against the pinned client's real HTTP call."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from phi_agents.rl import vllm_client as owner
from vllm import SamplingParams

from patch_loop_completion_boundary import patch
from owner_environment_transport import owner_sampling_params


def classes():
    path = Path(os.environ['LOOP_ORIGINAL_CLIENT'])
    source = path.read_text()
    assert '_completion_transport' not in source
    result = []
    for text in (source, patch(source)):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLLMClient')
        namespace = dict(vars(owner))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        result.append(namespace['VLLMClient'])
    assert patch(patch(source)) == patch(source)
    return result


class HTTPReply:
    status_code = 200
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def raise_for_status(self): pass
    def iter_content(self, **_):
        yield 'data: ' + json.dumps(dict(choices=[dict(text='response',
            finish_reason='stop', logprobs=dict(tokens=['token_id:21', 'token_id:22'],
                                              token_logprobs=[-.2, -.3]))]))
        yield 'data: [DONE]'


@pytest.mark.parametrize('kwargs', [dict(),
    dict(temperature=None, top_p=None, top_k=None, min_p=None,
         frequency_penalty=None, presence_penalty=None, repetition_penalty=None),
    dict(temperature=.7, top_p=.8, top_k=40, min_p=.02,
         frequency_penalty=.2, presence_penalty=.3, repetition_penalty=1.1,
         max_new_tokens=19, stop_token_ids=[31, 32], seed=42, logit_bias={'25': -.5})])
def test_default_and_callback_use_original_sampling_request(monkeypatch, kwargs):
    reference, candidate = classes()
    get = Mock(return_value=Mock(json=lambda: dict(data=[dict(id='base-model')])) )
    post = Mock(return_value=HTTPReply())
    monkeypatch.setattr(owner.requests, 'get', get)
    monkeypatch.setattr(owner.requests, 'post', post)
    expected = reference().get_completion(None, [7, 8], **kwargs)
    request = post.call_args.kwargs['json']
    assert candidate().get_completion(None, [7, 8], **kwargs) == expected
    assert post.call_args.kwargs['json'] == request
    client = candidate()
    client._completion_transport = Mock(return_value=expected)
    post.reset_mock(); get.reset_mock()
    assert client.get_completion(None, [7, 8], **kwargs) == expected
    carried = client._completion_transport.call_args.args[0]
    assert {k: v for k, v in carried.items() if k != 'model'} == {
        k: v for k, v in request.items() if k != 'model'}
    post.assert_not_called(); get.assert_not_called()
    sampling = {k: v for k, v in carried.items() if k not in ('prompt', 'model', 'stream')}
    # Exercise the installed native constructor, not a request recording fixture.
    actual = owner_sampling_params(SamplingParams(), [sampling], None)[0]
    assert actual == SamplingParams.from_optional(**sampling)


def test_cancelled_callback_keeps_original_cancellation(monkeypatch):
    _, candidate = classes()
    cancelled = Event(); cancelled.set()
    client = candidate(cancellation_event=cancelled)
    client._completion_transport = Mock()
    monkeypatch.setattr(owner.requests, 'post', Mock())
    assert client.get_completion('model', [7]) == ('', [], [], False, True)
    client._completion_transport.assert_not_called()
    owner.requests.post.assert_not_called()
