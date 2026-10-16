"""Reuse the existing native environment timeout for observed startup readiness."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
BASE = ROOT / 'candidates/appworld-environment-restart-boundary-20261017-v3/loop-owner'
VERSION = 'appworld-environment-readiness-20261017-v4'
DEST = ROOT / 'candidates' / VERSION / 'loop-owner'
AUDIT = ROOT / 'receipts' / VERSION
assert not DEST.exists()
measurement = json.loads((AUDIT / 'native-startup-measurement.json').read_bytes())
assert measurement['ready_count'] == 64 and measurement['above_original_five_second_window'] == 64
assert not any(x['alive'] for x in measurement['cleanup'])
name = 'phi_agents/appworld/interface.py'
old = (BASE / name).read_text()
expected = 'cd96661f67f65fa758d0aa46ddd031ac10884e0aa50f84679e4d3e51cb10d542'
assert hashlib.sha256(old.encode()).hexdigest() == expected
before = 'self._max_wait_tries: int = 5'
after = 'self._max_wait_tries: int = timeout_seconds'
assert old.count(before) == 1
new = old.replace(before, after)
original_tree, new_tree = ast.parse(old), ast.parse(new)
for node in ast.walk(new_tree):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute) and node.target.attr == '_max_wait_tries':
        assert isinstance(node.value, ast.Name) and node.value.id == 'timeout_seconds'
        node.value = ast.Constant(value=5)
assert ast.dump(original_tree) == ast.dump(new_tree)
DEST.mkdir(parents=True)
for source in BASE.iterdir():
    target = DEST / source.name
    if source.name == 'data':
        target.symlink_to(source, target_is_directory=True)
    elif source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.git', '.pytest_cache'))
    else:
        shutil.copy2(source, target)
(DEST / name).write_text(new)
evals = DEST / 'phi_agents/evals/appworld_evals.py'
record = dict(status='prepared_owner_patch_behavior_test_pending', version=VERSION, prepared=time.time(),
              upstream_commit='f14107a976e5793990329d3193df4742076c5a1d', baseline=str(BASE), owner_root=str(DEST),
              integration_baseline=str(ROOT/'candidates/direct-action-target-20261007-v3/appworld/loop-owner'),
              changes={name:dict(before_SHA256=expected, after_SHA256=hashlib.sha256(new.encode()).hexdigest())},
              patched_file=str(evals), after_SHA256=hashlib.sha256(evals.read_bytes()).hexdigest(),
              observed_failure=str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-DTPO-v3-detected-exit.json'),
              startup_measurement=str(AUDIT/'native-startup-measurement.json'),
              defect='Native readiness restarts a live service after about five seconds; all 64 original services took 6.14 to 7.19 seconds to become ready without restart',
              readiness_wait='Existing timeout_seconds (default100) one-second polls per native startup attempt',
              original_HTTP_timeout_seconds=100, original_max_restarts_on_error=2,
              native_inference_retry_budget_unchanged=True, sampler_reward_DT_PPO_unchanged=True,
              model_calls=0, optimizer_updates=0, existing_assets_reused=True)
(AUDIT/'owner-patch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
