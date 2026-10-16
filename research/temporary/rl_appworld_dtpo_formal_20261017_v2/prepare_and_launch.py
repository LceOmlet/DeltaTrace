"""Reuse the formal launcher with only the verified native AppWorld recovery fix."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
ORIGINAL_VERSION = 'appworld-dtpo-reward-target-exponential-formal-20261017-v1'
NEW_VERSION = 'appworld-dtpo-reward-target-exponential-formal-20261017-v2'
source = ROOT / 'receipts' / ORIGINAL_VERSION / 'source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a8c973eaef28c21b57762aa082e9be862c5ca53a82d8ddfb710e31ffa05a7a2c'
FIX_ROOT = ROOT / 'receipts/appworld-environment-initialize-retry-20261017-v2'
FIX = json.loads((FIX_ROOT / 'owner-patch.json').read_bytes())
assert json.loads((FIX_ROOT / 'CPU-recovery-verification.json').read_bytes())['status'] == 'passed'
assert hashlib.sha256(Path(FIX['patched_file']).read_bytes()).hexdigest() == FIX['after_SHA256']
tree = ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace(ORIGINAL_VERSION, NEW_VERSION)

prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
for index, node in enumerate(prepare.body):
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'env':
        prepare.body[index:index] = ast.parse('''
metadata_source = AUDIT / 'source/record_owner_binding.py'
shutil.copyfile(metadata_source, entry / 'record_owner_binding.py')
transport = entry / 'loop_owner_worker.py'
original_transport = transport.read_text()
anchor = "        output.put(('ready', rank))"
assert original_transport.count(anchor) == 1
transport.write_text(original_transport.replace(anchor,
    "        from record_owner_binding import record_owner_binding\\n        record_owner_binding(rank)\\n" + anchor))
original_tree, transport_tree = ast.parse(original_transport), ast.parse(transport.read_text())
for parent in ast.walk(transport_tree):
    if hasattr(parent, 'body') and isinstance(parent.body, list):
        parent.body = [child for child in parent.body if not
            (isinstance(child, ast.ImportFrom) and child.module == 'record_owner_binding') and not
            (isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
             and isinstance(child.value.func, ast.Name) and child.value.func.id == 'record_owner_binding')]
assert ast.dump(original_tree) == ast.dump(transport_tree)
''').body
        break
else:
    raise AssertionError('Original environment composition seam not found')

for index, node in enumerate(prepare.body):
    if isinstance(node, ast.Expr) and ast.unparse(node).startswith("env.pop('DT_LOOP_OVERRIDES_JSON'"):
        prepare.body[index + 1:index + 1] = ast.parse('''
env = {key:value.replace(FIX['baseline'], FIX['owner_root']) if isinstance(value,str) else value
       for key,value in env.items()}
''').body
        break
else:
    raise AssertionError('Original formal workload seam not found')

for index, node in enumerate(prepare.body):
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'bindings':
        prepare.body[index + 1:index + 1] = ast.parse('''
bindings = {path.replace(FIX['baseline'], FIX['owner_root']):digest for path,digest in bindings.items()}
bindings[str(entry/'loop_owner_worker.py')] = sha(entry/'loop_owner_worker.py')
bindings[str(entry/'record_owner_binding.py')] = sha(entry/'record_owner_binding.py')
for relative in ('phi_agents/evals/appworld_evals.py','phi_agents/appworld/interface.py','phi_agents/rl/appworld_scenario_runner.py'):
    actual = Path(FIX['owner_root'])/relative
    bindings[str(actual)] = sha(actual)
assert bindings[FIX['patched_file']] == FIX['after_SHA256']
prior = read(ROOT/'runs'/ORIGINAL_VERSION/'launch.json')
def updated_paths(value):
    if isinstance(value,str):
        return value.replace(ORIGINAL_VERSION,NEW_VERSION).replace(FIX['baseline'],FIX['owner_root'])
    if isinstance(value,dict):
        return {key:updated_paths(item) for key,item in value.items()}
    if isinstance(value,list):
        return [updated_paths(item) for item in value]
    return value
expected = updated_paths(prior['options'])
assert options == expected, 'Only run paths and verified environment owner may change'
''').body
        break
else:
    raise AssertionError('Original source binding seam not found')

for index, node in enumerate(prepare.body):
    if isinstance(node, ast.Expr) and ast.unparse(node).startswith("write(DEST / 'prepared.json'"):
        prepare.body[index:index] = ast.parse('''
record['environment_owner_patch'] = FIX
record['previous_failed_run'] = str(ROOT/'runs'/ORIGINAL_VERSION)
record['method_changed'] = False
record['training_numeric_options_unchanged'] = True
''').body
        break
else:
    raise AssertionError('Original prepared provenance write not found')

exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), globals())
