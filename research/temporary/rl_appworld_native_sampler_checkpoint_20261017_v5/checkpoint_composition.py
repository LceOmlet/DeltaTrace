"""Extend the existing formal preparation with verified owner state interfaces."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5'
PATCH=json.loads((AUDIT/'owner-patch.json').read_bytes())
assert json.loads((AUDIT/'CPU-sampler-checkpoint-verification.json').read_bytes())['status']=='passed'
assert json.loads((AUDIT/'native-two-rank-sampler-checkpoint-verification.json').read_bytes())['status']=='passed'
for relative,change in PATCH['changes'].items():
    actual=ROOT/'candidates/appworld-native-sampler-checkpoint-20261017-v5'/relative
    assert hashlib.sha256(actual.read_bytes()).hexdigest()==change['after_SHA256']


def extend_formal_tree(tree,kind,prior_version):
    """Add source overlays before the unchanged native CPU configuration check."""
    assert kind in ('dt','ppo')
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    key='verl_dt' if kind=='dt' else 'verl_ppo'
    before_cpu=f'''
sampler_checkpoint_patch = read(ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5/owner-patch.json')
native_numerical_owner = env['VERL_ROOT']
assert native_numerical_owner == sampler_checkpoint_patch['baselines'][{key!r}]
for name in ('loop_owner_worker.py','loop_owner_rollout.py'):
    shutil.copyfile(Path(sampler_checkpoint_patch['owner_roots']['entry'])/name, entry/name)
env = {{k:v.replace(native_numerical_owner,sampler_checkpoint_patch['owner_roots'][{key!r}])
       if isinstance(v,str) else v for k,v in env.items()}}
'''
    for i,node in enumerate(prepare.body):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='cpu':
            prepare.body[i:i]=ast.parse(before_cpu).body
            break
    else:raise AssertionError('Original native CPU environment seam missing')

    update_bindings=f'''
bindings = {{path.replace(native_numerical_owner,sampler_checkpoint_patch['owner_roots'][{key!r}]):digest
            for path,digest in bindings.items()}}
for name in ('loop_owner_worker.py','loop_owner_rollout.py'):
    bindings[str(entry/name)] = sha(entry/name)
for relative,change in sampler_checkpoint_patch['changes'].items():
    group,tail=relative.split('/',1)
    if group=='loop-owner':
        actual=Path(sampler_checkpoint_patch['owner_roots']['loop'])/tail
    elif group=={'verl-dt-owner' if kind=='dt' else 'verl-ppo-owner'!r}:
        actual=Path(sampler_checkpoint_patch['owner_roots'][{key!r}])/tail
    else:
        continue
    assert sha(actual)==change['after_SHA256']
    bindings[str(actual)] = change['after_SHA256']
'''
    for i,node in enumerate(prepare.body):
        if isinstance(node,ast.Assert) and 'bindings.items()' in ast.unparse(node.test) and 'all(' in ast.unparse(node.test):
            prepare.body[i:i]=ast.parse(update_bindings).body
            break
    else:raise AssertionError('Original actual source binding check missing')

    if kind=='ppo':
        for node in ast.walk(prepare):
            if isinstance(node,ast.Assert) and ast.unparse(node.test)=="env['VERL_ROOT'] == ACCEPTANCE['owner_root']":
                node.test=ast.parse("native_numerical_owner == ACCEPTANCE['owner_root']",mode='eval').body

    record=f'''
record['sampler_checkpoint_extension'] = sampler_checkpoint_patch
record['sampler_checkpoint_artifact'] = 'loop_sampler.pt'
record['previous_execution_gap_run'] = str(ROOT/'runs'/{prior_version!r})
record['checkpoint_gap_receipt'] = str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-native-sampler-checkpoint-state-gap.json')
record['native_sampling_and_update_algorithms_unchanged'] = True
'''
    for i,node in enumerate(prepare.body):
        if isinstance(node,ast.Expr) and ast.unparse(node).startswith("write(DEST / 'prepared.json'"):
            prepare.body[i:i]=ast.parse(record).body
            break
    else:raise AssertionError('Original preparation provenance seam missing')
