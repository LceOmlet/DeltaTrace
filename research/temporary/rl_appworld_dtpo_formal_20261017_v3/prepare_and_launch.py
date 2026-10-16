"""Reuse unchanged DTPO v5 formal composition with the verified environment owner."""
import ast
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-dtpo-reward-target-exponential-formal-20261017-v2/source/prepare_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8e258b1bc1f9896a1db4a6a15f0491e3afc2cc952cf8e3a36f1cc5989f6553a4'
tree=ast.parse(source.read_text())
for node in ast.walk(tree):
 if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
 node.value=node.value.replace('appworld-dtpo-reward-target-exponential-formal-20261017-v2',
                               'appworld-dtpo-reward-target-exponential-formal-20261017-v3')
 node.value=node.value.replace('appworld-environment-initialize-retry-20261017-v2',
                               'appworld-environment-restart-boundary-20261017-v3')
 node.value=node.value.replace("FIX['baseline']","FIX['integration_baseline']")
 if "record['environment_owner_patch'] = FIX" in node.value:
  node.value=node.value.replace("record['environment_owner_patch'] = FIX",
     "record['environment_owner_patch'] = FIX\nrecord['previous_failed_current_run'] = str(ROOT/'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v2')\nrecord['previous_execution_failure'] = str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-DTPO-v2-restart-readiness-exit.json')\nrecord['execution_version_only'] = True")
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),globals())
