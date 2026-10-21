"""Continue the original observer after a pre-RPC rank-count adapter failure."""
import ast
from prepare_campaign import ROOT, AUDIT, VERSION, sha

wrapper = AUDIT / 'source/observe_initial_workers.py'
assert sha(wrapper) == '7552833b77951db675a163081f31e1dddc24b3284d9edb537c750a6ad770f555'
tree = ast.parse(wrapper.read_text())
assert isinstance(tree.body[-1], ast.Expr)
assert isinstance(tree.body[-1].value, ast.Call)
assert isinstance(tree.body[-1].value.func, ast.Name) and tree.body[-1].value.func.id == 'exec'
namespace = {'__name__': 'compose_original_observer_without_starting_it', '__file__': str(wrapper)}
exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), str(wrapper), 'exec'), namespace)
code = namespace['code']
before = "script=AUDIT/'source/inspect_native_workers.py'"
assert code.count(before) == 1
code = code.replace(before, "script=AUDIT/'source/inspect_native_workers_dp8.py'")
exec(compile(code, str(wrapper), 'exec'), globals())
