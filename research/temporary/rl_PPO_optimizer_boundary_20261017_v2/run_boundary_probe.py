"""Reuse the native boundary probe without retaining full gradient copies.

Only the removable diagnostic observer changes: store fingerprints instead of
two complete CPU gradient vectors per rank. Native update/clipping/AdamW stay
owned by the pinned VERL and Torch implementations.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/PPO-optimizer-boundary-20261017-v1/source/run_boundary_probe.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '1d2c8362b794be88818aac1247b7865b09eeaca79c89b217fb70567ef0ff5acc'
tree = ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('PPO-optimizer-boundary-20261017-v1', 'PPO-optimizer-boundary-20261017-v2')

bounded_observer = r"""
for replay_node in nodes:
    if not isinstance(replay_node, ast.ClassDef) or replay_node.name != 'ReplayCritic':
        continue
    update = next(n for n in replay_node.body if isinstance(n, ast.FunctionDef) and n.name == 'replay_update')
    measured = next(n for n in update.body if isinstance(n, ast.FunctionDef) and n.name == 'measured_native_step')
    current = next(n for n in measured.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'current')
    current.value = ast.parse('{k:p.grad for k,p in self.critic_module.named_parameters() if p.grad is not None}', mode='eval').body
    branch = next(n for n in measured.body if isinstance(n, ast.If) and ast.unparse(n.test) == "label == 'native_reference'")
    branch.body = ast.parse('self._reference_gradients.append(hashes)').body
    branch.orelse = ast.parse('''
ref = self._reference_gradients[len(captures)]
delta = dict(different_parameter_names=[name for name,value in hashes.items() if value != ref[name]],
             numerical_error_not_computed=True,
             scope='Only fingerprint comparison; existing full-vector error receipts remain unchanged')
''').body
"""
seam_node = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'seam')
seam_node.value.value = bounded_observer + seam_node.value.value
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), globals())
