"""Real Qwen3.5 tokenizer at the native LOOP model/environment entry.

HTTP completions alone are fixtures. Never instantiates weights or a server.
"""
import ast
from pathlib import Path
from unittest.mock import Mock

import pytest
from phi_agents.rl.llm import qwen_3 as owner
from phi_agents.rl.type_defs import SystemMessage, UserMessage
from phi_agents.rl.vllm_client import MaxSeqLenExceeded
from patch_loop_tokenizer_entry import patch
from test_qwen35_environment_entry_local import ASSETS


def arguments():
    return dict(host='127.0.0.1', port=1, base_model_path=ASSETS,
        model_id=None, temperature=1., max_new_tokens=1500, top_p=None,
        top_k=None, min_p=None, frequency_penalty=None, max_model_len=32768)


def patched_class():
    source = Path(owner.__file__).read_text(encoding='utf-8')
    fixed = patch(source)
    assert patch(fixed) == fixed
    classes = [next(n for n in ast.parse(s).body if isinstance(n, ast.ClassDef) and n.name == 'VLLMQwen3')
               for s in (source, fixed)]
    methods = [{n.name: ast.dump(n) for n in c.body if isinstance(n, ast.FunctionDef)} for c in classes]
    assert {name for name in methods[0] if methods[0][name] != methods[1][name]} == {
        '__init__', 'get_generation_prompt_tokens', '_tokenize_with_tokenizer'}
    namespace = dict(vars(owner))
    exec(compile(fixed, owner.__file__, 'exec'), namespace)
    return namespace['VLLMQwen3']


def test_original_id_mismatch_and_native_client_fix():
    with pytest.raises(AssertionError):
        owner.VLLMQwen3(**arguments())
    llm = patched_class()(**arguments())
    assert {key: value.id for key, value in llm.special_tokens.items()} == {
        'bom': 248045, 'eom': 248046, 'eot': 248044}
    for value in llm.special_tokens.values():
        assert llm.tokenizer.decode([value.id]) == value.content
    assert llm.tokenizer.decode(llm.generation_prompt_tokens) == '<|im_start|>assistant\n<think>\n'


def test_native_messages_preserve_generated_ids_and_observation_mask():
    llm = patched_class()(**arguments())
    content = 'Read inventory.\n</think>\n\n```python\nprint(1)\n```'
    tokens = llm.tokenizer.encode(content, add_special_tokens=False) + [llm.tokenizer.eos_token_id]
    probs = [-.25] * len(tokens)
    transport = Mock(return_value=('', tokens, probs, False, False))
    llm._vllm.get_completion = transport
    prompt = [SystemMessage('Use the environment.'), UserMessage('Read inventory.')]
    action = llm.generate(prompt)
    assert action.generated_tokens == tokens and action.generated_token_logprobs == probs
    history = prompt + [action, UserMessage('1')]
    info = llm.get_policy_token_info(history)
    assert [t for t, selected in zip(info.tokens, info.is_output) if selected] == tokens
    assert [p for p, selected in zip(info.log_probs, info.is_output) if selected] == probs
    assert sum(info.is_output) == len(tokens)
    call = transport.call_args
    assert call.kwargs['max_new_tokens'] == 1500
    assert set(call.kwargs['stop_token_ids']) == {248044, 248045, 248046}
    assert call.args[1] == llm.get_tokens(prompt)[0] + list(llm.generation_prompt_tokens)


def test_native_context_budget_avoids_an_extra_generation_request():
    llm = patched_class()(**arguments())
    llm._vllm.get_completion = Mock()
    # Deterministic large input from real tokenizer, not a lucky rollout.
    history = [SystemMessage('Use tools.'), UserMessage('x ' * 33000)]
    assert len(llm.get_tokens(history)[0]) >= 32768
    with pytest.raises(MaxSeqLenExceeded):
        llm.generate(history)
    llm._vllm.get_completion.assert_not_called()


def test_formal_appworld_launcher_matches_native_workload(tmp_path, monkeypatch):
    import shutil
    from launch_appworld_native import options_for
    from loop_owner_recipe import compose
    from test_loop_environment_local import OWNER
    from test_official_workload_local import project_config, assert_project_trajectory_units
    monkeypatch.setenv('MODEL_PATH', str(ASSETS))
    monkeypatch.setenv('LOOP_ROOT', str(OWNER))
    # The native Hydra resolver reads the installed dev split to set the
    # evaluation batch. Use the author's exact asset in a temporary layout.
    datasets = tmp_path/'data/datasets'
    datasets.mkdir(parents=True)
    shutil.copyfile(OWNER/'data/appworld_splits/dev.txt', datasets/'dev.txt')
    monkeypatch.setenv('APPWORLD_ROOT', str(tmp_path))
    native = compose(OWNER, [])  # README command and native Hydra defaults.
    options, _ = options_for(tmp_path)
    cfg = project_config(options)
    assert_project_trajectory_units(cfg, groups=native.rl.params.scenarios_per_iteration,
        samples=native.rl.params.rollouts_per_scenario,
        global_mini=native.rl.params.minibatch_size, epochs=native.rl.params.epochs_per_iteration)
    assert cfg['trainer']['total_training_steps'] == native.rl.params.total_iterations
    assert cfg['trainer']['test_freq'] == native.rl.eval.eval_every_n_iterations
    assert cfg['env']['max_steps'] == native.rl.scenario_runner.appworld_config.env.max_interactions
