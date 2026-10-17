"""A32/B64/A32 within128 natural input preparations; automatically restore.

Only the prepared native owner's integer H2D copy branch differs in B. The
original model, sampler, async input-prep lifetime and native workload remain.
This does not launch additional generation or optimizer operations.
"""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('observe_natural_mrope_copy.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f543592f2a191721275db5c0d5c443f9a6ffa9994d4590bb7d469033872058b5'
code = source.read_text()


def replace(before, after):
    global code
    assert code.count(before) == 1, before
    code = code.replace(before, after)


replace("VERSION = 'native-natural-MRoPE-copy-observation-20261017-v1'",
        "VERSION = 'native-MRoPE-row-copy-ABA-20261017-v1'")
replace('    import hashlib\n    import inspect', '    import ast\n    import hashlib\n    import inspect\n    import sys')
replace("    delegate = runner._prepare_inputs\n", """    delegate = runner._prepare_inputs
    proof = json.loads((AUDIT/'native-MRoPE-integer-copy-20261017-v1-completed.json').read_bytes())
    assert len(proof['results']) == 8
    assert all(r['all_planned_sizes_completed'] and r['RNG_unchanged'] and all(
        c['baseline_full_buffer_equal_expected'] and c['candidate_full_buffer_equal_expected'] and
        c['full_buffer_pair_equal'] for c in r['cases']) for r in proof['results'])
    prepared = json.loads((AUDIT/'native-MRoPE-row-copy-v1-prepared.json').read_bytes())
    candidate_path = Path(prepared['candidate'])
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == prepared['candidate_SHA256']
    assert prepared['disabled_branch_whole_module_AST_equal_original']
    assert '_native_mrope_row_copy' not in vars(runner)
    tree = ast.parse(candidate_path.read_text())
    owner_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='GPUModelRunner')
    function = next(n for n in owner_class.body if isinstance(n, ast.FunctionDef) and n.name=='_prepare_inputs')
    assert not function.decorator_list and not any(isinstance(n,ast.Name) and n.id=='super' for n in ast.walk(function))
    namespace = dict(vars(sys.modules[type(runner).__module__]))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(candidate_path),'exec'),namespace)
    candidate = MethodType(namespace['_prepare_inputs'],runner)
""")
replace('    target_addresses = (positions.gpu.data_ptr(), positions.cpu.data_ptr())',
        """    target_addresses = {(positions.gpu.data_ptr()+axis*positions.gpu.stride(0)*positions.gpu.element_size(),
        positions.cpu.data_ptr()+axis*positions.cpu.stride(0)*positions.cpu.element_size()) for axis in range(3)}""")
replace('== target_addresses', 'in target_addresses')
replace("        additional_model_tensor_copy_optimizer_sync_calls=0, calls=[],",
        """        additional_model_tensor_copy_optimizer_sync_calls=0, calls=[],
        resource_probe='A32 original / B64 exact-int64 native row copy / A32 original',
        candidate_source=str(candidate_path),candidate_SHA256=prepared['candidate_SHA256'],
        disabled_branch_whole_module_AST_equal_original=True,""")
replace("        entry = dict(started=time.time(), position_copies=[])",
        """        index = len(record['calls'])
        arm = 'original' if index<32 or index>=96 else 'native_row_copy'
        scheduler = args[0] if args else kwargs['scheduler_output']
        entry = dict(started=time.time(), position_copies=[],arm=arm,
            scheduled_tokens=scheduler.total_num_scheduled_tokens,
            requests=len(scheduler.num_scheduled_tokens),
            new_requests=len(scheduler.scheduled_new_reqs))""")
replace('                return delegate(*args, **kwargs)',
        "                return (delegate if arm=='original' else candidate)(*args, **kwargs)")
replace("len(record['calls']) >= 64", "len(record['calls']) >= 128")
replace('                del runner._prepare_inputs\n',
        '                del runner._prepare_inputs\n                del runner._native_mrope_row_copy\n')
replace('    runner._prepare_inputs = MethodType(observe, runner)',
        '    runner._native_mrope_row_copy = True\n    runner._prepare_inputs = MethodType(observe, runner)')
exec(compile(code, str(source), 'exec'), globals())
