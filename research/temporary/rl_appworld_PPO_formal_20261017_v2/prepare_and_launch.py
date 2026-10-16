"""Reuse formal native PPO with the verified original environment restart-boundary fix."""
import ast
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-native-PPO-formal-20261017-v1/source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='68d4a601f65819066bb20dc1cfba3919cd3676131888377ae159fe217ed11de9'
tree=ast.parse(source.read_text())
for node in ast.walk(tree):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):
  node.value=node.value.replace('appworld-native-PPO-formal-20261017-v1','appworld-native-PPO-formal-20261017-v2')
composition=r'''
for current_node in ast.walk(tree):
    if isinstance(current_node,ast.Constant) and isinstance(current_node.value,str):
        current_node.value=current_node.value.replace('appworld-environment-initialize-retry-20261017-v2',
                                                      'appworld-environment-restart-boundary-20261017-v3')
        current_node.value=current_node.value.replace("FIX['baseline']", "FIX['integration_baseline']")
        if "record['environment_owner_patch'] = FIX" in current_node.value:
            current_node.value=current_node.value.replace("record['environment_owner_patch'] = FIX",
                "record['environment_owner_patch'] = FIX\nrecord['previous_failed_PPO_run'] = str(ROOT/'runs/appworld-native-PPO-formal-20261017-v1')\nrecord['previous_PPO_exit_receipt'] = str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-PPO-v1-restart-readiness-exit.json')")
'''
tree.body[-1:-1]=ast.parse(composition).body
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),globals())
