"""CPU-only contracts for selecting the saved residual and owner references.

No torch, model, environment, remote connection, or GPU computation is imported.
These tests establish diagnosis wiring/syntax only, not numerical acceptance.
"""
import ast
from pathlib import Path
import re
import subprocess

import pytest


DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[2]
LEASES = DIRECTORY/'diagnose_native_prefix_leases.py'
COMPONENTS = DIRECTORY/'diagnose_native_prefix_components.py'
STAGER = DIRECTORY/'run_native_prefix_reuse_workload.py'
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
    for path in (LEASES, COMPONENTS, STAGER):
        compile(tree(path), str(path), 'exec')


def test_current_peak_is_local_row_two_of_original_b4():
    records = [{'sorted_index':i} for i in range(88)]
    selected, local_row, consumer_index = pure_function(
        LEASES, 'select_component_request')(records, offset=40, limit=4, request_index=42)
    assert [row['sorted_index'] for row in selected] == [40,41,42,43]
    assert local_row == 2 and consumer_index == 10
    assert len(records) == 88


@pytest.mark.parametrize('phase_only',[False,True])
def test_shared_diagnostic_prepares_same_bank_once_for_multiple_consumers(phase_only):
    """Execute the real diagnostic closure without importing model/runtime."""
    body=function(tree(LEASES),'diagnose')
    initializer=next(node for node in body.body if isinstance(node,ast.Assign)
                     and any(isinstance(target,ast.Name) and target.id=='shared_bank'
                             for target in node.targets))
    factory=function(body,'shared_factory')
    wrapper=ast.FunctionDef(name='make_factory',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=name) for name in
            ('phase_only','all_requests','offset','limit','prepare_native_prefix_leases')],
            kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[initializer,factory,ast.Return(value=ast.Name(id='shared_factory',ctx=ast.Load()))],
        decorator_list=[])
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
         str(LEASES)+':shared_factory','exec'),namespace)
    requests=[object() for _ in range(88)]
    selected_requests=requests[40:44]
    leases=[object() for _ in range(22)]
    preparation=dict(capture_and_preparation_seconds=23.96,capture_rounds=5)
    calls=[]
    def owner_prepare(*args,**kwargs):
        calls.append((args,kwargs))
        return leases,preparation
    shared=namespace['make_factory'](phase_only,requests,40,4,owner_prepare)
    runner=object()
    first,first_report=shared(runner,selected_requests,minibatch_size=4,eos_token_id=99)
    second,second_report=shared(runner,selected_requests,minibatch_size=4,eos_token_id=99)
    third,third_report=shared(runner,selected_requests,minibatch_size=4,eos_token_id=99)
    assert len(calls)==1
    assert calls[0]==((runner,requests if phase_only else selected_requests),
                      dict(minibatch_size=4,eos_token_id=99))
    assert first is second is third
    if phase_only:
        assert first==leases[10:11]
    else:
        assert first is leases
    assert first_report['diagnostic_bank_reused'] is False
    assert second_report['diagnostic_bank_reused'] is third_report['diagnostic_bank_reused'] is True
    assert first_report['capture_and_preparation_seconds']==second_report['capture_and_preparation_seconds']==23.96
    assert preparation==dict(capture_and_preparation_seconds=23.96,capture_rounds=5)
    if phase_only:
        assert first_report['diagnostic_selected_consumer_batches']==1


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


def test_explicit_recorded_layer_skips_auto_probe_and_selects_native_family():
    body = function(tree(COMPONENTS),'diagnose')
    assert ast.unparse(assignment(body,'auto_layer').value) == (
        'gdn_layer_index is None and observed_layer_index is None')
    explicit = next(node for node in body.body if isinstance(node,ast.If)
                    and ast.unparse(node.test) == 'observed_layer_index is not None')
    assert ast.unparse(assignment(explicit,'attention_only').value) == (
        "not hasattr(layers[observed_layer_index], 'linear_attn')")
    assert not calls(explicit,'runner.forward_prefix')
    auto = next(node for node in body.body if isinstance(node,ast.If)
                and ast.unparse(node.test) == 'auto_layer')
    assert len(calls(auto,'output_probe')) == 2
    attention = next(node for node in body.body if isinstance(node,ast.If)
                     and ast.unparse(node.test) == 'attention_only')
    check, = calls(attention,'check_first_attention')
    assert {item.arg:ast.unparse(item.value) for item in check.keywords}['query_window_tokens'] == (
        '64 if bounded_operator else None')
    assert any(isinstance(node,ast.Return) for node in attention.body)
    fla_loop = next(node for node in body.body if isinstance(node,ast.For)
                    and isinstance(node.target,ast.Tuple)
                    and [ast.unparse(item) for item in node.target.elts] == ['label','case'])
    assert body.body.index(attention) < body.body.index(fla_loop)


def test_stager_passes_explicit_layer_without_changing_original_b4_selection():
    source = STAGER.read_text(encoding='utf8')
    assert "parser.add_argument('--component-layer',type=int," in source
    assert "args.component_layer is not None and (not args.components_only" in source
    assert 'component_layer=@COMPONENT_LAYER@' in source
    assert "run_env['DT_PREFIX_COMPONENT_LAYER']=str(component_layer)" in source
    assert "DT_PREFIX_COMPONENT_REQUEST_INDEX=str(peak['row'])" in source
    assert "DT_PREFIX_DIAGNOSTIC_ROWS='4'" in source
    assert "component_layer=component_layer" in source
    assert ".replace('@COMPONENT_LAYER@',repr(args.component_layer))" in source


def test_stager_profiles_only_requested_warm_b4_and_labels_instrumentation():
    source = STAGER.read_text(encoding='utf8')
    assert "parser.add_argument('--hot-profile',action='store_true'," in source
    assert 'args.hot_profile and not (args.phase_only and args.warm_phases)' in source
    assert "'hot-phase' if args.hot_profile" in source
    assert "run_env['DT_PREFIX_HOT_PROFILE']='1'" in source
    assert 'instrumented_hot_profile=@HOT_PROFILE@' in source
    assert ".replace('@HOT_PROFILE@',str(args.hot_profile))" in source


@pytest.mark.parametrize('components,layer,warm,hot',[(True,3,False,False),(False,None,True,True)])
def test_embedded_remote_python_compiles_for_both_bounded_modes(components, layer, warm, hot):
    module = tree(STAGER)
    scripts = [node.value for node in ast.walk(module) if isinstance(node,ast.Constant)
               and isinstance(node.value,str) and 'component_layer=@COMPONENT_LAYER@' in node.value]
    script, = scripts
    python = script.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    replacements = {
        '@COMPONENTS_ONLY@':repr(components), '@COMPONENT_LAYER@':repr(layer),
        '@BOUNDED_ONLY@':'True', '@WARM_PHASES@':repr(warm), '@HOT_PROFILE@':repr(hot),
    }
    for old, new in replacements.items():
        python = python.replace(old,new)
    python = re.sub(r'@[A-Z_]+@','fixture',python)
    compile(python,str(STAGER)+':remote-python','exec')


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
