"""Preserve class compilation context when binding the tested resource method.

The first binding client rejected a separately compiled AST method before any
mutation. Compiling its original complete module as data reproduces the actual
live bytecode exactly; no module imports or module body are executed here.
"""
import hashlib
from pathlib import Path
import sys
import types

legacy = Path(__file__).resolve().parent / 'install_DT_native_host_lifetime_TextCraft_v1.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '42c3a7f2fd0d1de932d506edd38707d6ecc5cb9db2fadcc475b61a22e4895c8c'
code = legacy.read_text()
old = """    def owned_code(path):
        parsed = ast.parse(path.read_bytes())
        c, = [c for c in parsed.body if isinstance(c, ast.ClassDef) and c.name == cls.name]
        fn, = [f for f in c.body if isinstance(f, ast.FunctionDef) and f.name == 'attribute']
        fn = copy.deepcopy(fn)
        fn.decorator_list = []
        namespace = dict(raw.__globals__)
        namespace['os'] = os
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec', dont_inherit=True), namespace)
        return namespace['attribute'].__code__
"""
new = """    def owned_code(path):
        import types
        compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
        class_code, = [x for x in compiled.co_consts
                       if isinstance(x, types.CodeType) and x.co_name == cls.name]
        function_code, = [x for x in class_code.co_consts
                          if isinstance(x, types.CodeType) and x.co_name == 'attribute']
        return function_code
"""
scan_old = """    modules = {id(m): m for m in list(sys.modules.values()) if m is not None
               and getattr(m, '__file__', None)
               and Path(m.__file__).resolve() == original_path.resolve()}
"""
scan_new = """    modules = {id(m): m for m in list(sys.modules.values()) if m is not None
               and isinstance(getattr(m, '__file__', None), str)
               and Path(m.__file__).name == original_path.name
               and Path(m.__file__).resolve() == original_path.resolve()}
"""
for before, after in [
    (old, new), (scan_old, scan_new),
    ("VERSION = 'DT-native-host-lifetime-TextCraft-20261017-v1'",
     "VERSION = 'DT-native-host-lifetime-TextCraft-20261017-v2'"),
]:
    assert code.count(before) == 1
    code = code.replace(before, after, 1)

if '--preflight' in sys.argv:
    import json
    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    facts = json.loads((root / 'receipts/dt-native-replay-host-lifetime-20261017-v3/prepared.json').read_bytes())
    result = []
    for key in ('original_root', 'candidate_root'):
        path = Path(facts[key]) / facts['changed_sources'][0]
        expected = facts['original_SHA256' if key == 'original_root' else 'candidate_SHA256']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
        cls, = [x for x in compiled.co_consts if isinstance(x, types.CodeType) and x.co_name == 'Qwen35DenseFiniteRunner']
        fn, = [x for x in cls.co_consts if isinstance(x, types.CodeType) and x.co_name == 'attribute']
        result.append(dict(path=str(path), source_SHA256=expected,
                           code_SHA256=hashlib.sha256(fn.co_code).hexdigest(), freevars=list(fn.co_freevars)))
    assert result[0]['code_SHA256'] == '7ef304de915e3e9434331907b3683031aff20540f89af0de1b5837e0c9512b07'
    assert all(x['freevars'] == [] for x in result)
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='CPU_preflight_full_module_compile_matches_actual_live_method',
                          results=result, model_calls=0, CUDA_initialized=False,
                          module_bodies_executed=False, binding_calls=0)))
    raise SystemExit(0)

exec(compile(code, str(legacy), 'exec'), globals())
