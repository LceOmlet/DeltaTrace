"""Compose native AppWorld PPO only after the completed owner acceptance receipt."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
ACCEPTANCE = json.loads((ROOT / 'receipts/PPO-native-final-numerical-20261017-v1/formal-acceptance.json').read_bytes())
assert ACCEPTANCE['status'] == 'accepted_for_formal_AppWorld_PPO'
source = ROOT / 'receipts/appworld-dtpo-reward-target-exponential-formal-20261017-v2/source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '8e258b1bc1f9896a1db4a6a15f0491e3afc2cc952cf8e3a36f1cc5989f6553a4'
tree = ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('appworld-dtpo-reward-target-exponential-formal-20261017-v2',
                                        'appworld-native-PPO-formal-20261017-v1')
        node.value = node.value.replace("record['previous_failed_run']", "record['environment_failure_reference']")
        if 'expected = updated_paths' in node.value:
            node.value = node.value.replace('expected = updated_paths(prior[\'options\'])',
                "expected = updated_paths(prior['options'])\nexpected.update({key:value for key,value in baseline['options'].items() if key.startswith(('critic.', '+critic.'))})\nexpected['algorithm.adv_estimator'] = 'gae'\nexpected['trainer.max_critic_ckpt_to_keep'] = 2")

native_PPO_composition = r"""
for native_node in ast.walk(tree):
    if not isinstance(native_node, ast.Constant):
        continue
    if native_node.value == 'candidates/appworld-sql-native-speed-20261016-v3/AppWorld/prepared.json':
        native_node.value = 'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json'
    elif native_node.value == 'dt':
        native_node.value = 'ppo'
    elif native_node.value == 'deltatrace':
        native_node.value = 'gae'
    elif native_node.value == '4545c55807e7d002287717b6a4254f9b083dd5fc':
        native_node.value = '20bd331bdbc9026a5668e11362178e10ab7400c8'
prepare = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
for index,node in enumerate(prepare.body):
    if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='options':
        prepare.body[index+1:index+1] = ast.parse('''
options.update({key:value for key,value in baseline['options'].items() if key.startswith(('critic.', '+critic.'))})
options['algorithm.adv_estimator'] = 'gae'
options['trainer.max_critic_ckpt_to_keep'] = 2
assert env['VERL_ROOT'] == ACCEPTANCE['owner_root']
''').body
        break
else:
    raise AssertionError('Native author options seam not found')
for index,node in enumerate(prepare.body):
    if isinstance(node,ast.Expr) and ast.unparse(node).startswith("write(DEST / 'prepared.json'"):
        prepare.body[index:index] = ast.parse('''
record['accepted_PPO_receipt'] = ACCEPTANCE
record['memory_owner_patch_commit'] = '1bb6520308d2ca0eea19b010922c9cfba3d8d130'
record['budget']['critic_checkpoints_to_keep'] = 2
''').body
        break
for native_node in ast.walk(tree):
    if isinstance(native_node,ast.Constant) and native_node.value == '6,7':
        native_node.value = '2,3'
    if isinstance(native_node,ast.List) and len(native_node.elts)==2 and all(isinstance(x,ast.Constant) for x in native_node.elts) and [x.value for x in native_node.elts]==[6,7]:
        native_node.elts = [ast.Constant(2),ast.Constant(3)]
    if isinstance(native_node,ast.Constant) and isinstance(native_node.value,str):
        native_node.value = native_node.value.replace('(?:6|7)', '(?:2|3)').replace('Physical GPU6/7 occupied','Physical GPU2/3 occupied')
"""
tree.body[-1:-1] = ast.parse(native_PPO_composition).body
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), globals())
