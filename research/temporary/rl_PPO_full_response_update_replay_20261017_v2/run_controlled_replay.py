"""Reuse native replay with test-only deterministic Conv1d and actual operands.

No PPO, optimizer or owner implementation changes. Original native update,
state restore and observers are reused from the pinned v1 diagnostic. The
Conv1d flag is for this numerical reference only, never production or timing.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
base = ROOT / 'receipts/PPO-full-response-update-replay-20261017-v1/source/run_update_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == '2117d945e21a1e9d4b0b665e611fa204aa7b298870b3a68d821afaf8b67432b5'
tree = ast.parse(base.read_text())
old_version = 'PPO-full-response-update-replay-20261017-v1'
new_version = 'PPO-full-response-update-replay-20261017-v2/execution2'
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace(old_version, new_version)

seam = r'''
for replay_node in nodes:
    if isinstance(replay_node, ast.ClassDef) and replay_node.name == 'ReplayCritic':
        initialize = next(n for n in replay_node.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_replay')
        for i, statement in enumerate(initialize.body):
            if isinstance(statement, ast.Expr) and ast.unparse(statement).startswith('super().init_model()'):
                initialize.body[i + 1:i + 1] = ast.parse('from controlled_reference_worker import install_actual_conv_capture\ninstall_actual_conv_capture(self, AUDIT)').body
                break
        else:
            raise AssertionError('Native initialization seam not found')
        returned = next(n for n in initialize.body if isinstance(n, ast.Return))
        returned.value.keywords.append(ast.keyword(arg='actual_conv_binding', value=ast.parse('self._actual_conv_binding', mode='eval').body))
'''
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and node.value == 'metax_fsdp_event_wait.worker_setup':
        node.value = 'controlled_reference_worker.worker_setup'
for node in tree.body:
    if isinstance(node, ast.If) and ast.unparse(node.test) == "'--native' not in sys.argv":
        node.body.insert(-1, ast.parse("env['PYTHONPATH'] = str(AUDIT / 'source') + ':' + env['PYTHONPATH']").body[0])
    if isinstance(node, ast.If) and ast.unparse(node.test) == "'--preflight' in sys.argv":
        node.body.insert(0, ast.parse('ray.cloudpickle.dumps(ReplayCritic.__ray_metadata__.modified_class)').body[0])
for index, node in enumerate(tree.body):
    if isinstance(node, ast.Expr) and ast.unparse(node).startswith('exec(compile('):
        tree.body[index:index] = ast.parse(seam).body
        break
else:
    raise AssertionError('Pinned native replay class loading not found')
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'status' for x in node.targets):
        node.value.keywords.append(ast.keyword(arg='test_only_cudnn_deterministic', value=ast.Constant(True)))
exec(compile(ast.fix_missing_locations(tree), str(base), 'exec'), globals())
