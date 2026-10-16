"""Reuse accepted native PPO with the verified sampler checkpoint extension."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sys.path.insert(0,str(ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5/source'))
from checkpoint_composition import extend_formal_tree
source=ROOT/'receipts/appworld-native-PPO-formal-20261017-v1/source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='68d4a601f65819066bb20dc1cfba3919cd3676131888377ae159fe217ed11de9'
tree=ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
        node.value=node.value.replace('appworld-native-PPO-formal-20261017-v1',
                                      'appworld-native-PPO-formal-20261017-v3')
composition=r'''
for current_node in ast.walk(tree):
    if isinstance(current_node,ast.Constant) and isinstance(current_node.value,str):
        current_node.value=current_node.value.replace('appworld-environment-initialize-retry-20261017-v2',
                                                      'appworld-native-sampler-checkpoint-20261017-v5')
        current_node.value=current_node.value.replace("FIX['baseline']","FIX['integration_baseline']")
tree.body[-1:-1]=ast.parse("extend_formal_tree(tree,'ppo','appworld-native-PPO-formal-20261017-v2')").body
'''
tree.body[-1:-1]=ast.parse(composition).body
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),globals())
