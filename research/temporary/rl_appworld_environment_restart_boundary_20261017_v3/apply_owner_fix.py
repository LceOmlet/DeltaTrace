"""Keep observed restart readiness failures inside the existing LOOP retry budget."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
BASE=ROOT/'candidates/appworld-environment-initialize-retry-20261017-v2/loop-owner'
VERSION='appworld-environment-restart-boundary-20261017-v3'
DEST=ROOT/'candidates'/VERSION/'loop-owner'
AUDIT=ROOT/'receipts'/VERSION
assert not DEST.exists()
files={
 'phi_agents/appworld/interface.py':'17adc301aae395b789b5528abc167fd025b6cbb22379e954c600ed586987610e',
 'phi_agents/evals/appworld_evals.py':'c1a71a8f7860dd0cb192293229429c4024405e1b0f6070d34bc183b1edd05676',
}
changes={}
for name,expected in files.items():
 old=(BASE/name).read_text()
 assert hashlib.sha256(old.encode()).hexdigest()==expected
 if name.endswith('interface.py'):
  before='raise RuntimeError(f"Could not connect to {self._remote_environment_url}")'
  after='raise AppWorldInitializeTaskError(f"Could not connect to {self._remote_environment_url}")'
  assert old.count(before)==1
  new=old.replace(before,after)
  original_tree,new_tree=ast.parse(old),ast.parse(new)
  for node in ast.walk(new_tree):
   if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call) and isinstance(node.exc.func,ast.Name) and node.exc.func.id=='AppWorldInitializeTaskError':
    if ast.unparse(node.exc.args[0])=='f\'Could not connect to {self._remote_environment_url}\'':node.exc.func.id='RuntimeError'
 else:
  before='            world.restart()\n            continue'
  after='''            try:
                world.restart()
            except AppWorldInitializeTaskError:
                logger.exception("AppWorld restart readiness failed within the current inference retry budget")
            continue'''
  assert old.count(before)==1
  new=old.replace(before,after)
  original_tree,new_tree=ast.parse(old),ast.parse(new)
  for node in ast.walk(new_tree):
   if isinstance(node,ast.ExceptHandler) and isinstance(node.type,ast.Tuple) and len(node.body)>1 and isinstance(node.body[1],ast.Try):
    nested=node.body[1]
    assert len(nested.handlers)==1 and ast.unparse(nested.handlers[0].type)=='AppWorldInitializeTaskError'
    node.body[1:2]=nested.body
 assert ast.dump(original_tree)==ast.dump(new_tree),name
 changes[name]=dict(before_SHA256=expected,after_SHA256=hashlib.sha256(new.encode()).hexdigest(),text=new)
DEST.mkdir(parents=True)
for source in BASE.iterdir():
 target=DEST/source.name
 if source.name=='data':target.symlink_to(source,target_is_directory=True)
 elif source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','.git','.pytest_cache'))
 else:shutil.copy2(source,target)
for name,change in changes.items():(DEST/name).write_text(change.pop('text'))
AUDIT.mkdir(parents=True,exist_ok=True)
record=dict(status='prepared_owner_patch_behavior_test_pending',version=VERSION,prepared=time.time(),
 upstream_commit='f14107a976e5793990329d3193df4742076c5a1d',baseline=str(BASE),owner_root=str(DEST),changes=changes,
 integration_baseline=str(ROOT/'candidates/direct-action-target-20261007-v3/appworld/loop-owner'),
 patched_file=str(DEST/'phi_agents/evals/appworld_evals.py'),after_SHA256=changes['phi_agents/evals/appworld_evals.py']['after_SHA256'],
 defect='Original world.restart readiness exception escaped the existing inference retry branch and propagates through future.result/get_rollouts',
 evidence=str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-PPO-current-error-contexts.json'),
 original_retry_budget_unchanged=True,original_timeouts_unchanged=True,original_sampler_reward_DT_PPO_unchanged=True,
 model_calls=0,optimizer_updates=0,existing_assets_reused=True)
(AUDIT/'owner-patch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
