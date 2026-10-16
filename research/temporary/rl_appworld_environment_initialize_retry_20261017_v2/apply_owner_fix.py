"""Extend the existing LOOP crash recovery for the observed initialization timeout."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
BASE = ROOT / 'candidates/direct-action-target-20261007-v3/appworld/loop-owner'
VERSION = 'appworld-environment-initialize-retry-20261017-v2'
DEST = ROOT / 'candidates' / VERSION / 'loop-owner'
AUDIT = ROOT / 'receipts' / VERSION
FILE = Path('phi_agents/evals/appworld_evals.py')
assert not DEST.exists(), 'Keep the previous source and receipts immutable'
old = (BASE / FILE).read_text()
assert hashlib.sha256(old.encode()).hexdigest() == 'd51913602b21a26702055670e72ed155cf24f0894b478e6eeb33e74451182435'
assert old.count('    AppWorldExecutionError,\n') == 1
assert old.count('except (AppWorldExecutionError, requests.HTTPError, requests.ConnectionError):') == 1
new = old.replace('    AppWorldExecutionError,\n',
                  '    AppWorldExecutionError,\n    AppWorldInitializeTaskError,\n')
new = new.replace('except (AppWorldExecutionError, requests.HTTPError, requests.ConnectionError):',
                  'except (AppWorldExecutionError, AppWorldInitializeTaskError, requests.HTTPError, requests.ConnectionError, requests.Timeout):')
before, after = ast.parse(old), ast.parse(new)
for tree in (before, after):
    # Only these two additions may differ from the native owner.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'phi_agents.appworld.interface':
            node.names = [n for n in node.names if n.name != 'AppWorldInitializeTaskError']
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
            node.type.elts = [n for n in node.type.elts
                             if ast.unparse(n) not in ('AppWorldInitializeTaskError', 'requests.Timeout')]
assert ast.dump(before) == ast.dump(after)
DEST.mkdir(parents=True)
for source in BASE.iterdir():
    target = DEST / source.name
    if source.name == 'data':
        target.symlink_to(source, target_is_directory=True)
    elif source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.git', '.pytest_cache'))
    else:
        shutil.copy2(source, target)
(DEST / FILE).write_text(new)
AUDIT.mkdir(parents=True, exist_ok=True)
record = dict(version=VERSION, prepared_unix=time.time(), upstream_commit='f14107a976e5793990329d3193df4742076c5a1d',
              baseline=str(BASE), owner_root=str(DEST), patched_file=str(DEST / FILE),
              before_SHA256=hashlib.sha256(old.encode()).hexdigest(),
              after_SHA256=hashlib.sha256(new.encode()).hexdigest(),
              changes=['Observed wrapped initialize failure joins original restart/retry branch',
                       'Observed underlying requests timeout joins the same branch'],
              retry_loop_and_budget_unchanged=True, timeout_unchanged=True,
              task_reward_DT_PPO_unchanged=True, existing_task_assets_reused=True,
              status='prepared_owner_patch_behavior_test_pending')
(AUDIT / 'owner-patch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
