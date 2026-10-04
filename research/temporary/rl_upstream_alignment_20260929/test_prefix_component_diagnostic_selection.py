"""CPU-only contracts for selecting the saved residual and owner references.

No torch, model, environment, remote connection, or GPU computation is imported.
These tests establish diagnosis wiring/syntax only, not numerical acceptance.
"""
import ast
from pathlib import Path
import subprocess

import pytest


DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[2]
LEASES = DIRECTORY/'diagnose_native_prefix_leases.py'
COMPONENTS = DIRECTORY/'diagnose_native_prefix_components.py'
BASELINE = '265c41a04bc70752617745846d222894641f3fe4'


def tree(path):
    return ast.parse(path.read_text(encoding='utf8'), filename=str(path))


def function(module, name):
    return next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def pure_function(path, name):
    namespace = {}
    node = function(tree(path), name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def baseline_tree(path):
    relative = path.relative_to(REPO).as_posix()
    source = subprocess.check_output(['git','show',f'{BASELINE}:{relative}'],
                                     cwd=REPO, text=True, encoding='utf8')
    return ast.parse(source)


def assignment(module, name):
    return next(node for node in ast.walk(module)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))


def calls(module, expression):
    return [node for node in ast.walk(module) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == expression]


def test_diagnostic_files_compile_without_importing_runtime():
    for path in (LEASES, COMPONENTS):
        compile(tree(path), str(path), 'exec')


def test_current_peak_is_local_row_two_of_original_b4():
    records = [{'sorted_index':i} for i in range(88)]
    selected, local_row, consumer_index = pure_function(
        LEASES, 'select_component_request')(records, offset=40, limit=4, request_index=42)
    assert [row['sorted_index'] for row in selected] == [40,41,42,43]
    assert local_row == 2 and consumer_index == 10
    assert len(records) == 88


@pytest.mark.parametrize('offset,limit,index',[(41,4,42),(40,1,42),(0,4,42),(88,4,88)])
def test_selection_rejects_another_geometry(offset, limit, index):
    with pytest.raises(ValueError):
        pure_function(LEASES,'select_component_request')(
            list(range(88)), offset=offset, limit=limit, request_index=index)


def test_component_capture_passes_complete_bank_and_real_lease_rows():
    body = function(tree(LEASES),'diagnose')
    branch = next(node for node in body.body if isinstance(node,ast.If)
                  and 'DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC' in ast.unparse(node.test))
    preparation, = calls(branch,'prepare_native_prefix_leases')
    assert [ast.unparse(arg) for arg in preparation.args] == ['runner','all_requests']
    assert ast.unparse(assignment(branch,'lease').value) == 'leases[batch_index]'
    source = next(node for node in ast.walk(branch) if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Tuple) and
                          [ast.unparse(item) for item in target.elts] == ['source','source_row']
                          for target in node.targets))
    assert ast.unparse(source.value) == 'lease.sources[local_row]'
    assert ast.unparse(assignment(branch,'prefix').value) == 'lease.prefix_length'
    component, = calls(branch,'components')
    keywords = {item.arg:ast.unparse(item.value) for item in component.keywords}
    assert keywords['matched_rows'] == '[(source_row, local_row)]'
    assert keywords['gdn_layer_index'] == 'None'
    assert keywords['operand_output_dir'] == 'Path(out)'


def test_first_unequal_layer_is_observed_and_not_hardcoded():
    select = pure_function(COMPONENTS,'first_changed_layer')
    assert select([{'layer':i,'equal':i != 7} for i in range(32)]) == 7
    assert select([{'layer':i,'equal':True} for i in range(32)]) is None


def test_fsdP_scheduling_uses_original_min_collective_before_any_early_return():
    module = tree(COMPONENTS)
    synchronize = function(module,'synchronize_observation_layer')
    collective, = calls(synchronize,'torch.distributed.all_reduce')
    assert {item.arg:ast.unparse(item.value) for item in collective.keywords} == {
        'op':'torch.distributed.ReduceOp.MIN', 'group':'group'}
    assert ast.unparse(assignment(synchronize,'value').value) == 'layer_count if selected is None else selected'
    diagnose = function(module,'diagnose')
    auto = next(node for node in diagnose.body if isinstance(node,ast.If)
                and ast.unparse(node.test) == 'auto_layer')
    collective_assignment = assignment(auto,'selected')
    early = next(node for node in auto.body if isinstance(node,ast.If)
                 and ast.unparse(node.test) == 'selected is None')
    assert auto.body.index(collective_assignment) < auto.body.index(early)
    assert 'own_selected' in ast.unparse(collective_assignment.value)


def test_original_fla_assertion_cannot_use_ci_warning_escape_hatch():
    body = function(tree(COMPONENTS),'diagnose')
    assert any(isinstance(node,ast.Assert) and
               ast.unparse(node.test) == 'not owner.FLA_CI_ENV' for node in body.body)


def test_original_reference_selection_and_assertion_execution_are_unchanged():
    previous = baseline_tree(COMPONENTS)
    current = tree(COMPONENTS)
    for name in ('diagnose','check_first_attention'):
        old, new = function(previous,name), function(current,name)
        assert ast.dump(assignment(old,'assertions')) == ast.dump(assignment(new,'assertions'))
        compiled = 'original_assertions' if name == 'diagnose' else 'original'
        assert ast.dump(assignment(old,compiled)) == ast.dump(assignment(new,compiled))
    for expression in ('reference.recurrent_gated_delta_rule_ref',
                       "namespace['attention_ref']"):
        assert [ast.dump(node) for node in calls(previous,expression)] == [
            ast.dump(node) for node in calls(current,expression)]


def test_automatic_attention_reference_uses_actual_tail_and_full_prefix_kv():
    check = function(tree(COMPONENTS),'check_first_attention')
    assert 'query_start:prefix' in ast.unparse(assignment(check,'q').value)
    kv = next(node for node in ast.walk(check) if isinstance(node,ast.Assign)
              and any(isinstance(target,ast.Tuple) and
                      [ast.unparse(item) for item in target.elts] == ['k','v']
                      for target in node.targets))
    assert '[:, :prefix]' in ast.unparse(kv.value)
    assert 'query_start:prefix' in ast.unparse(assignment(check,'out').value)
