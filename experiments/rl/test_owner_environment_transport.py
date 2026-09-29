"""Transport tests against the pinned, verified VERL/vLLM classes.

Generation is a recording fixture here. This tests row/ID/mask contracts, not
model numerics; the latter retain their separate official-method receipts.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as owner

from patch_verl_environment_entry import patch_vllm, patch_collector, patch_main
from owner_environment_transport import owner_sampling_params, policy_reply, preprocess_owner_tokens
from test_vllm_active_rows import make_prompts, make_rollout


@pytest.fixture(autouse=True)
def cpu_transport_test_has_no_gpu_telemetry(monkeypatch):
    from verl.utils.debug import performance
    # Only the decorator's GPU telemetry is replaced in this CPU transport test.
    # Production sources and all tested owner data transformations stay intact.
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0., 0., 0., 0.))


def patched_class():
    source = Path(owner.__file__).read_text()
    fixed = patch_vllm(source)
    assert patch_vllm(fixed) == fixed
    cls = next(n for n in ast.parse(fixed).body if isinstance(n, ast.ClassDef) and n.name == 'vLLMRollout')
    namespace = dict(vars(owner))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), owner.__file__, 'exec'), namespace)
    return namespace['vLLMRollout']


def baseline_class():
    # Keep the reference independent when tests run inside the entry candidate.
    path=Path(os.environ['VERL_ENTRY_BASELINE'])
    source=path.read_text()
    assert 'owner_sampling_kwargs' not in source, 'Reference already contains the new seam'
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='vLLMRollout')
    namespace=dict(vars(owner))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['vLLMRollout']


@pytest.mark.parametrize('active', [None, [True]*4, [False, True, False, True], [False]*4])
@pytest.mark.parametrize('mode', ['sample', 'greedy', 'validate'])
def test_default_owner_route_is_identical(active, mode):
    baseline = make_rollout(baseline_class())
    candidate = make_rollout(patched_class())
    expected = baseline.generate_sequences(make_prompts(active, mode=mode))
    actual = candidate.generate_sequences(make_prompts(active, mode=mode))
    for key in expected.batch.keys():
        torch.testing.assert_close(actual.batch[key], expected.batch[key], rtol=0, atol=0)
    assert actual.non_tensor_batch.keys() == expected.non_tensor_batch.keys()


class Engine:
    def __init__(self):
        self.calls = []

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params))
        outputs = []
        for i, (prompt, params) in enumerate(zip(prompts, sampling_params)):
            # Stop-token is deliberately not the tokenizer EOS (7); also exercise
            # stop strings and max-length without any EOS, exactly as vLLM allows.
            ids = [31, 32, 9] if i == 0 else [41, 42]
            reason = 'stop' if i == 0 else 'length'
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=ids, text=f'raw engine text {i}', finish_reason=reason,
                logprobs=[{t: SimpleNamespace(logprob=-.25)} for t in ids])]))
        return outputs


def test_custom_stops_keep_exact_lengths_tokens_logprobs_and_row_order():
    rollout = make_rollout(patched_class())
    rollout.inference_engine = Engine()
    batch = make_prompts([False, True, False, True])
    batch.non_tensor_batch['owner_sampling_kwargs'] = np.array([
        dict(max_tokens=3, stop_token_ids=[9], skip_special_tokens=False),
        dict(max_tokens=3, stop_token_ids=[9], skip_special_tokens=False),
        dict(max_tokens=2), dict(max_tokens=2)], dtype=object)
    output = rollout.generate_sequences(batch)
    assert output.non_tensor_batch['owner_response_length'].tolist() == [0, 3, 0, 2]
    assert output.batch['attention_mask'][:, -1024:].sum(-1).tolist() == [0, 3, 0, 2]
    assert policy_reply(output, 1).token_ids == [31, 32, 9]
    assert policy_reply(output, 1).logprobs == [-.25]*3
    assert policy_reply(output, 1).text == 'raw engine text 0'
    assert policy_reply(output, 1).finish_reason == 'stop'
    assert policy_reply(output, 3).finish_reason == 'length'
    assert len(rollout.inference_engine.calls[0][0]) == 2
    assert rollout.sampling_params.max_tokens == 1024


def test_sampling_uses_official_constructor_stop_cache_and_validation():
    base = SamplingParams(logprobs=0, temperature=.6, max_tokens=3000)
    changes = dict(stop=['</sql>', '</solution>'], stop_token_ids=[9],
                   include_stop_str_in_output=True, skip_special_tokens=False)
    actual = owner_sampling_params(base, [changes], None)[0]
    expected = SamplingParams(logprobs=0, temperature=.6, max_tokens=3000, **changes)
    assert actual == expected
    assert 9 in actual.all_stop_token_ids
    with pytest.raises(ValueError):
        owner_sampling_params(base, [dict(temperature=-1)], None)


def test_raw_prompt_ids_use_verl_padding_without_template_or_retokenization():
    class NoTextTokenizer:
        pad_token_id = 0

        def __getattr__(self, name):
            raise AssertionError(f'Raw IDs must not call tokenizer.{name}')

    collector = SimpleNamespace(tokenizer=NoTextTokenizer(),
        config=SimpleNamespace(data=SimpleNamespace(max_prompt_length=8)))
    batch = SimpleNamespace(non_tensor_batch={'data_source': ['sql']})
    obs = dict(raw_prompt_ids=[[7, 81, 19]], sampling_kwargs=[dict(max_tokens=3)])
    result = preprocess_owner_tokens(collector, 0, batch, obs)
    assert result['raw_prompt_ids'] == [7, 81, 19]
    assert result['input_ids'].tolist() == [0]*5 + [7, 81, 19]
    assert result['attention_mask'].tolist() == [0]*5 + [1]*3
    assert result['position_ids'][-3:].tolist() == [0, 1, 2]
    obs['raw_prompt_ids'] = [list(range(9))]
    with pytest.raises(NotImplementedError):
        preprocess_owner_tokens(collector, 0, batch, obs)


def test_entry_patches_are_default_inert_and_idempotent():
    import agent_system.multi_turn_rollout.rollout_loop as collector
    import verl.trainer.main_ppo as main
    for module, patch in ((collector, patch_collector), (main, patch_main)):
        source = Path(module.__file__).read_text()
        fixed = patch(source)
        assert patch(fixed) == fixed
        compile(fixed, module.__file__, 'exec')
