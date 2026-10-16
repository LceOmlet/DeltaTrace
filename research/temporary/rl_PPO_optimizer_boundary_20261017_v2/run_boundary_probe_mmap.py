"""Reuse the native boundary probe without retaining full gradient copies.

Store fingerprints instead of full diagnostic gradient copies and use the
official Torch mmap IO option for the existing warm checkpoint. Native VERL
still restores and verifies all state; native update/clipping/AdamW stay intact.
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
    replay_node.body += ast.parse('''
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
    checkpoint_root = Path(local_path)
    assert checkpoint_root.name == 'numerical-warm-state-original-VERL'
    native_load = torch.load
    calls = []
    def native_mmap_load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).is_relative_to(checkpoint_root):
            kwargs['mmap'] = True
            calls.append(str(file))
        return native_load(file, *args, **kwargs)
    torch.load = native_mmap_load
    try:
        return super().load_checkpoint(local_path, hdfs_path=hdfs_path,
                                       del_local_after_load=del_local_after_load)
    finally:
        torch.load = native_load
        self._native_mmap_restores = getattr(self, '_native_mmap_restores', []) + [calls]
        write(AUDIT/f'native-mmap-restores-rank{self.rank}.json',
              dict(calls=self._native_mmap_restores,
                   scope='Only official Torch mmap storage IO for the existing diagnostic checkpoint; native VERL restores all state'))
''').body
"""
seam_node = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'seam')
seam_node.value.value = bounded_observer + seam_node.value.value
rename_preflight = ast.parse('''
for preflight_node in ast.walk(tree):
    if isinstance(preflight_node, ast.Constant) and preflight_node.value == 'CPU-preflight.json':
        preflight_node.value = 'CPU-preflight-mmap.json'
''').body
tree.body[-1:-1] = rename_preflight
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), globals())
