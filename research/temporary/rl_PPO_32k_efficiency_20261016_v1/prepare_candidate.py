"""Minimal owner extension: active text groups and existing non-reentrant checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='PPO-32k-native-efficiency-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION/'verl-owner'
BASE=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(),'Preserve an existing candidate'
layout=json.loads((AUDIT/'actual-critic-layout.json').read_bytes())
for rank in layout['results']:
    assert rank['model_type']=='FSDPQwen3_5ForTokenClassification'
    assert rank['first_parameter_dtype']=='torch.float32'
    assert sum('visual.blocks' in node['name'] for node in rank['wrapped_nodes'])==27
    assert sum('language_model.layers' in node['name'] for node in rank['wrapped_nodes'])==32
test=json.loads((AUDIT/'native-offload-group-reproduction.json').read_bytes())
assert test['original_handler_full_model_count']['error']['type']=='AssertionError'
assert test['original_handler_active_text_count']['error'] is None
authorities={str(ROOT/name):sha(ROOT/name) for name in ('formal-training.json','active-training.json','active-source.json')}
shutil.copytree(BASE,DEST,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
changes=[]
p=DEST/'verl/utils/activation_offload.py'
assert sha(p)=='829042503f87abeeff2e4c86b27b409acb04d71b503820bbb5fa80eb7bd1d950'
before=p.read_text()
edits=[
 ('def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):',
  'def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt, use_reentrant=True):'),
 ('                use_reentrant=True,','                use_reentrant=use_reentrant,'),
 ('def enable_activation_offloading(model, strategy, enable_ckpt=False):',
  'def enable_activation_offloading(model, strategy, enable_ckpt=False, use_reentrant=True):'),
 ('handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)',
  'handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt, use_reentrant=use_reentrant)'),
]
after=before
for old,new in edits:
    assert after.count(old)==1
    after=after.replace(old,new)
restored=after
for old,new in reversed(edits):restored=restored.replace(new,old)
assert restored==before
compile(after,str(p),'exec');p.write_text(after)
changes.append(dict(path=str(p),before=sha(BASE/'verl/utils/activation_offload.py'),after=sha(p),
    scope='Expose the existing Torch checkpoint use_reentrant option; native default True, actor path unchanged'))
p=DEST/'verl/workers/fsdp_workers.py'
assert sha(p)=='9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
before=p.read_text()
old='            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)'
new='''            if config.strategy == "fsdp2" and critic_model_config.model_type == "qwen3_5":
                # Text-only PPO executes language_model layers. Counting the
                # inactive vision branch breaks the original offloader's group
                # prefetch schedule. Preserve the critic's original Torch
                # non-reentrant checkpoint and its early-stop recomputation.
                enable_activation_offloading(
                    critic_module.model.language_model, config.strategy,
                    enable_gradient_checkpointing, use_reentrant=False)
            else:
''' + '    '+old
assert before.count(old)==1
after=before.replace(old,new)
assert after.replace(new,old)==before
compile(after,str(p),'exec');p.write_text(after)
changes.append(dict(path=str(p),before=sha(BASE/'verl/workers/fsdp_workers.py'),after=sha(p),
    scope='Only the existing optional critic activation-offload path for Qwen3.5 FSDP2: select active language subtree, keep non-reentrant checkpoint'))
unchanged={name:sha(DEST/name) for name in ('verl/workers/critic/dp_critic.py','verl/workers/actor/dp_actor.py',
    'verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/utils/experimental/torch_functional.py')}
assert all(value==sha(BASE/name) for name,value in unchanged.items())
assert authorities=={p:sha(p) for p in authorities}
record=dict(version=VERSION,status='prepared_only_not_deployed',base=str(BASE),VERL_ROOT=str(DEST),
    changes=changes,unchanged_owner_sources=unchanged,formal_authorities_unchanged=authorities,
    caller_configuration_change='critic.model.enable_activation_offload=True only in independent verification candidate',
    model_dtype='Original CPU FP32 masters, BF16 forward, FP32 reduce unchanged',
    original_checkpoint_mode='use_reentrant=False; original critic non-reentrant early-stop preserved',
    model_calls=0,DT_calls=0,optimizer_updates=0)
(AUDIT/'prepared-candidate.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
