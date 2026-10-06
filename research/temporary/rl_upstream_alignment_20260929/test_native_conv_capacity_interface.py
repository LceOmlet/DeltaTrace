"""CPU scope and input contracts for the isolated native-conv capacity mode.

Reuse the existing capacity test's exact helper and tokenizer double. These
tests neither load Qwen nor replace the recorded-real-source GPU diagnosis.
"""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import types

import pytest

from test_capacity_fixture_readout_interface import (
    REPO, TokenizerDouble, actual_helper, original_row,
)


SOURCE = Path(__file__).with_name('diagnose_native_prefix_leases.py')


def source_nodes():
    module = ast.parse(SOURCE.read_bytes())
    diagnose = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'diagnose')
    prepare = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_prepare_native_conv_capacity_inputs')
    verify = next(node for node in diagnose.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'verify_prepare')
    expected = next(node for node in diagnose.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'expected'
                            for target in node.targets))
    guard = next(node for node in ast.walk(diagnose) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Tuple)
                 and any(isinstance(target, ast.Name) and target.id == 'expected_source_sha256'
                         for target in node.target.elts))
    return prepare, verify, expected, guard


def test_capacity_scope_bindings_do_not_overwrite_original_owners():
    prepare, verify, expected, guard = source_nodes()
    assert [target.id for target in guard.target.elts] == [
        'name', 'value', 'expected_source_sha256']
    protected = {'runner', 'readout', 'requests', 'native_prepare', 'expected'}
    loop_targets = {target.id for node in ast.walk(prepare)
                    if isinstance(node, (ast.For, ast.comprehension))
                    for target in ast.walk(node.target) if isinstance(target, ast.Name)}
    assert not protected.intersection(loop_targets)
    assert ast.unparse(expected.value).startswith("{(r['traj_uid'], r['source_step']): r")
    assert {node.id for node in ast.walk(verify) if isinstance(node, ast.Name)} >= {
        'expected', 'requests', 'native_prepare', 'conv_capacity',
        'capacity_episodes', 'capacity_factual_inputs', 'capacity_returns'}


def capacity_contract(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(REPO/'experiments/rl'))
    from reward_readout import EventRatioReadout
    readout = EventRatioReadout(object(), TokenizerDouble(), task='AppWorld',
        max_steps=40, packed_answer_targets=object(), appworld_num_tests=2,
        sampling=dict(temperature=1., max_tokens=1500))
    rows = [{**original_row(torch, step=25), 'traj_uid': f'cpu-source-{index}'}
            for index in range(4)]
    returns = [1., .5, 1., .5]
    report = dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
    _, requests = readout._prepare_episode(rows, report, returns)
    helper_module = types.ModuleType('verify_dt_context_capacity')
    helper_module.capacity_fixture = actual_helper(torch)
    monkeypatch.setitem(sys.modules, helper_module.__name__, helper_module)
    prepare, _, _, _ = source_nodes()
    namespace = dict(torch=torch, Path=Path, hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prepare], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    result = namespace[prepare.name](readout, requests, dict(rows=rows), tmp_path, 0)
    return torch, readout, requests, result


def test_capacity_inputs_call_existing_helper_and_prepare_exactly(monkeypatch, tmp_path):
    torch, readout, _, result = capacity_contract(monkeypatch, tmp_path)
    episodes, returns, requests, factuals, details, receipt = result
    assert len(episodes) == len(returns) == len(requests) == len(factuals) == len(details) == 4
    assert len({episode[0]['traj_uid'] for episode in episodes}) == 4
    assert receipt['sha256'] == hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    for index, request in enumerate(requests):
        assert request['actions'].numel() == 512
        assert request['context_tokens'] == details[index]['dt_input_tokens'] == 32768
        prepared = torch.cat([request[key] for key in ('prompt','actions','query','target')])
        assert torch.equal(prepared, factuals[index])
        assert torch.equal(prepared, saved['factual_inputs'][index])
        assert hashlib.sha256(prepared.numpy().tobytes()).hexdigest() == details[index]['factual_input_ids_sha256']
        assert request['observed_return'] == returns[index][0]
    assert readout.max_steps == 40 and readout.sampling == dict(temperature=1., max_tokens=1500)


def make_actual_verify(namespace, requests, *, capacity, result):
    """Execute the source closure with its actual source guard target bindings.

    Only the guard's file-read body/iterator is replaced with three sentinels;
    this is a Python binding test, not an imported owner/hash verification.
    """
    _, verify, expected, guard = source_nodes()
    binding_guard = ast.For(target=copy.deepcopy(guard.target),
        iter=ast.parse("[('runner', None, 'runner-sha'), ('finite', None, 'finite-sha'), ('model', None, 'model-sha')]").body[0].value,
        body=[ast.Pass()], orelse=[])
    setup = ast.parse('''
def make_verify(native_prepare, requests, conv_capacity, capacity_episodes,
                capacity_factual_inputs, capacity_returns):
    pass
''').body[0]
    setup.body = [copy.deepcopy(expected), copy.deepcopy(verify), binding_guard,
                  ast.Return(value=ast.Tuple(elts=[ast.Name(id='verify_prepare', ctx=ast.Load()),
                                                  ast.Name(id='expected', ctx=ast.Load())], ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setup], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    episodes, returns, _, factuals, _, _ = result
    return namespace['make_verify'](namespace['readout']._prepare_episode, requests,
                                    capacity, episodes, factuals, returns)


def test_original_verify_closure_keeps_request_mapping_in_both_branches(monkeypatch, tmp_path):
    torch, readout, original_requests, result = capacity_contract(monkeypatch, tmp_path)
    namespace = dict(torch=torch, readout=readout)
    capacity_requests = result[2]
    capacity_verify, capacity_expected = make_actual_verify(namespace, capacity_requests,
        capacity=True, result=result)
    assert isinstance(capacity_expected, dict)
    assert 'expected' in capacity_verify.__code__.co_freevars
    for episode, returns in zip(result[0], result[1]):
        report = dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
        _, pending = capacity_verify(episode, report, returns)
        assert len(pending) == 1
    original_verify, original_expected = make_actual_verify(namespace, original_requests,
        capacity=False, result=result)
    rows = [{**original_row(torch, step=25), 'traj_uid': f'cpu-source-{index}'}
            for index in range(4)]
    report = dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
    _, pending = original_verify(rows, report, [1., .5, 1., .5])
    assert len(pending) == 4 and isinstance(original_expected, dict)
    assert set(original_expected) == {(r['traj_uid'], r['source_step']) for r in original_requests}
