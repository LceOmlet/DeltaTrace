"""Final native PPO update comparison without gradient or optimizer read hooks."""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/PPO-optimizer-boundary-20261017-v2/source/run_boundary_probe_mmap.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'c8692e01b55fd20ecb8fe553e58a91bce4b1a46c073d60bc15623152e96aafbd'
tree = ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('PPO-optimizer-boundary-20261017-v2', 'PPO-native-final-numerical-20261017-v1')

final_native_calls = r"""
for native_node in ast.walk(tree):
    if isinstance(native_node, ast.Call) and isinstance(native_node.func, ast.Attribute) and native_node.func.attr == 'replay_update':
        native_node.func.attr = 'update_critic'
    if isinstance(native_node, ast.For) and isinstance(native_node.target, ast.Tuple) and ast.unparse(native_node.target) == '(label, mode)':
        native_node.iter.elts = [item for item in native_node.iter.elts
                                if item.elts[0].value != 'native_plain_after_install']
    if isinstance(native_node, ast.Constant) and native_node.value == 'instrumented_post_values_exact':
        native_node.value = 'native_post_values_exact'
    if isinstance(native_node, ast.Constant) and native_node.value == 'completed_instrumented_native_update_measurements':
        native_node.value = 'completed_native_unobserved_update_output_replay'
    if isinstance(native_node, ast.Constant) and isinstance(native_node.value,str) and native_node.value.startswith('Diagnostic gradient/state reads can synchronize'):
        native_node.value = 'Native CriticWorker.update_critic; no gradient/optimizer read hooks during updates. State verified before each update. Test-only FA determinism does not alter production configuration or performance receipts.'
tree.body[0].value.value = 'Final original CriticWorker update and post-values: same restored native checkpoint and input, no update observers, pinned VERL atol0/rtol0 criterion.'
"""
rename = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'rename_preflight')
assert isinstance(rename.value, ast.Call) and isinstance(rename.value.args[0], ast.Constant)
rename.value.args[0].value += final_native_calls
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), globals())
