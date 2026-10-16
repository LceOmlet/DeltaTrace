"""Keep the original HF/FSDP checkpoint call and compose native offload around it."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='PPO-32k-native-efficiency-20261016-v2'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION/'verl-owner'
BASE=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(),'Preserve an existing candidate'
prior=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit-vs-candidate-fit.json'
result=json.loads(prior.read_bytes())
assert result['numerical_status']=='failed' and result['compared']['native_values']
AUDIT.mkdir(exist_ok=True)
shutil.copytree(BASE,DEST,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
changes=[]
p=DEST/'verl/utils/activation_offload.py'
assert sha(p)=='829042503f87abeeff2e4c86b27b409acb04d71b503820bbb5fa80eb7bd1d950'
before=p.read_text()
old='def enable_activation_offloading(model, strategy, enable_ckpt=False):'
new='def enable_activation_offloading(model, strategy, enable_ckpt=False, preserve_hf_checkpoint=False):'
assert before.count(old)==1
after=before.replace(old,new)
anchor='    if enable_ckpt:\n        # The implementation of activation checkpointing in transformers library is incompatible with activation offloading,'
branch='''    if preserve_hf_checkpoint:
        assert enable_ckpt, "Preserving HF checkpoint requires the native checkpoint path"
        # The HF checkpoint function wraps the original module __call__,
        # including FSDP hooks. Keep that function and its native kwargs intact.
        # Only checkpoint inputs are saved in this outer offload context;
        # Torch's own checkpoint hook owns tensors inside the original layer.
        handler = ActivationHandler(context, sync_func, tensor_filter, False)

        def wrap_native_checkpoint(module):
            native_checkpoint = module._gradient_checkpointing_func

            @functools.wraps(native_checkpoint)
            def checkpoint_with_offload(*args, **kwargs):
                with context:
                    return handler.forward(module, native_checkpoint, *args, **kwargs)

            module._gradient_checkpointing_func = checkpoint_with_offload

        for layer in layers:
            assert layer.gradient_checkpointing and hasattr(layer, "_gradient_checkpointing_func")
            wrap_native_checkpoint(layer)
        return

'''
assert after.count(anchor)==1
after=after.replace(anchor,branch+anchor)
assert after.replace(branch,'').replace(new,old)==before
compile(after,str(p),'exec');p.write_text(after)
changes.append(dict(path=str(p),before=sha(BASE/'verl/utils/activation_offload.py'),after=sha(p),
    scope='Default-inert owner composition of native async offload around the exact existing HF checkpoint function; no replacement checkpoint, scheduler or tensor filter'))
p=DEST/'verl/workers/fsdp_workers.py'
assert sha(p)=='9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
before=p.read_text()
old='            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)'
new='''            if config.strategy == "fsdp2" and critic_model_config.model_type == "qwen3_5":
                enable_activation_offloading(
                    critic_module.model.language_model, config.strategy,
                    enable_gradient_checkpointing, preserve_hf_checkpoint=True)
            else:
'''+'    '+old
assert before.count(old)==1
after=before.replace(old,new)
assert after.replace(new,old)==before
compile(after,str(p),'exec');p.write_text(after)
changes.append(dict(path=str(p),before=sha(BASE/'verl/workers/fsdp_workers.py'),after=sha(p),
    scope='Optional Qwen3.5 FSDP2 critic activation offload only; active text subtree and untouched native HF checkpoint'))
unchanged={name:sha(DEST/name) for name in ('verl/workers/critic/dp_critic.py','verl/workers/actor/dp_actor.py',
    'verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/utils/experimental/torch_functional.py')}
assert all(value==sha(BASE/name) for name,value in unchanged.items())
record=dict(version=VERSION,status='prepared_only_unaccepted',VERL_ROOT=str(DEST),base=str(BASE),changes=changes,
    unchanged_owner_sources=unchanged,previous_candidate_unaccepted=dict(path=str(prior),SHA256=sha(prior)),
    reason='First candidate moved the checkpoint from HF __call__ to forward. Preserve the exact native callable before attributing observed backward differences.',
    native_checkpoint_mode='Original existing partial and all kwargs preserved; no checkpoint packing replacement',
    original_B4_32k_GAE_PPO_optimizer_precision_preserved=True,formal_mutations=0,model_calls=0,optimizer_updates=0)
(AUDIT/'prepared-candidate.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
