"""Reuse DTPO v5 formal composition with verified native sampler checkpoints."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sys.path.insert(0,str(ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5/source'))
from checkpoint_composition import extend_formal_tree
source=ROOT/'receipts/appworld-dtpo-reward-target-exponential-formal-20261017-v2/source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8e258b1bc1f9896a1db4a6a15f0491e3afc2cc952cf8e3a36f1cc5989f6553a4'
tree=ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
        node.value=node.value.replace('appworld-dtpo-reward-target-exponential-formal-20261017-v2',
                                      'appworld-dtpo-reward-target-exponential-formal-20261017-v5')
        node.value=node.value.replace('appworld-environment-initialize-retry-20261017-v2',
                                      'appworld-native-sampler-checkpoint-20261017-v5')
        node.value=node.value.replace("FIX['baseline']","FIX['integration_baseline']")
tree.body[-1:-1]=ast.parse("extend_formal_tree(tree,'dt','appworld-dtpo-reward-target-exponential-formal-20261017-v4')").body
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),globals())
