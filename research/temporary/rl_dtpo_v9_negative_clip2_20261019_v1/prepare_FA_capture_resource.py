"""Prepare a storage-only patch against the actually imported native owner.

Do not change the default DT_ROOT, imported methods, or any training process.
The candidate reuses the owner's GPU capture destination and native finite
operations while keeping GDN and decoder-gate host storage unchanged.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

CODE = r'''
import ast,difflib,hashlib,json,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=R/'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
before=owner.read_text();after=before
changes=[
 ("            mixer_device='cpu' if offload_mixer else 'cuda'",
  "            mixer_device='cpu' if offload_mixer else 'cuda'\n"
  "            if is_fa and observer is None and getattr(self,'fa_replay_capture_cuda',False):\n"
  "                mixer_device='cuda'"),
 ("destination=mixer_device,copy_tensors=copy_captures,retained_names=attention_needed,preserve_strides=offload_mixer,pinned_host=offload_mixer and self.pin_replay_host)",
  "destination=mixer_device,copy_tensors=copy_captures,retained_names=attention_needed,preserve_strides=offload_mixer,pinned_host=offload_mixer and self.pin_replay_host and mixer_device=='cpu')"),
 ("if isinstance(value,torch.Tensor) and name not in dense_aliases:",
  "if isinstance(value,torch.Tensor) and name not in dense_aliases and not value.is_cuda:")]
for old,new in changes:
 assert after.count(old)==1,old
 after=after.replace(old,new)
tree=ast.parse(after)
cls,=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen35DenseFiniteRunner']
fn,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='attribute']
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='super' for n in ast.walk(fn))
assert not any(isinstance(n,ast.Attribute) and n.attr.startswith('__') for n in ast.walk(fn))
compile(tree,str(owner),'exec')
old_tree=ast.parse(before);old_cls,=[n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen35DenseFiniteRunner']
unchanged=[]
for node in old_tree.body:
 if isinstance(node,ast.FunctionDef):
  other,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==node.name]
  assert ast.dump(node,include_attributes=False)==ast.dump(other,include_attributes=False);unchanged.append(node.name)
for node in old_cls.body:
 if isinstance(node,ast.FunctionDef) and node.name!='attribute':
  other,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==node.name]
  assert ast.dump(node,include_attributes=False)==ast.dump(other,include_attributes=False);unchanged.append('Qwen35DenseFiniteRunner.'+node.name)
C=R/'candidates/dt-FA-capture-GPU-original-gate-host-20261019-v1'
C.mkdir(parents=True,exist_ok=True)
target=C/'qwen35_dense_finite_runner.py';diff=C/'native-owner-resource.patch'
if target.exists():assert target.read_text()==after
else:target.write_text(after)
patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(owner),tofile=str(target)))
diff.write_text(patch)
record=dict(status='prepared_only_original_owner_resource_patch_not_deployed',collected=time.time(),
 native_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
 candidate=dict(path=str(target),SHA256=hashlib.sha256(target.read_bytes()).hexdigest()),
 patch=dict(path=str(diff),SHA256=hashlib.sha256(diff.read_bytes()).hexdigest(),text=patch),
 unchanged_functions=unchanged,default_flag=False,resource_flag='fa_replay_capture_cuda',
 actual_training_modified=False,default_launch_modified=False,DT_ROOT_changed=False,
 new_model_calls=0,new_DT_calls=0,new_optimizer_calls=0,
 scope='Only the original FA capture destination, matching pinned-host flag, and skip of already-CUDA copies. FA/GDN finite arithmetic, decoder gate destination, masks, Q/V/A, original cache consumption and PPO remain unchanged.',
 numerical_verification='Not performed. AST/source checks are not FA/FLA/VERL numerical acceptance.',
 efficiency_or_capacity_verified=False)
(C/'prepared.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''

record = json.loads(run_python(CODE, timeout=40))
target = Path('experiments/rl/results_dtpo_v9_negative2_FA_capture_resource_prepared_20261019_v4.json')
target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'],candidate=record['candidate'],patch=record['patch']['text']),ensure_ascii=False))
