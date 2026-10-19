"""Isolate the v9 credit seam; preserve the six accepted native owners."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'
MODE='reward_target_exponential_clipped_v9'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('v9_credit_patch_owner',AUDIT/'source/patch_verl_agent2.py')
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            parent=ROOT/'receipts'/f'{task}-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8/{method}-prepared.json'
            if task=='textcraft' and method=='dt':
                parent=ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/dt-prepared.json'
            row=json.loads(parent.read_bytes())
            for path,expected in row['source_bindings'].items():assert sha(path)==expected,path
            entry=Path(row['entry']);verl=Path(row['verl_root']);old_entry=entry;old_verl=verl
            changes={};env=dict(row['effective_environment']);bindings=dict(row['source_bindings'])
            if method=='dt':
                candidate=ROOT/'candidates'/f'{task}-dt-v9-clip-u-20261019-v1'
                candidate.mkdir(exist_ok=False)
                entry=candidate/'entry';verl=candidate/'verl-owner'
                ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc')
                shutil.copytree(old_entry,entry,ignore=ignore);shutil.copytree(old_verl,verl,ignore=ignore)
                for name in ('counterfactual.py','dt_training_batch.py','reward_readout.py','textcraft_owner_rollout.py'):
                    source=AUDIT/'source'/name
                    ast.parse(source.read_text())
                    changes[str(entry/name)]=dict(before=sha(old_entry/name),after=sha(source))
                    shutil.copyfile(source,entry/name)
                trainer=verl/'verl/trainer/ppo/ray_trainer.py'
                before=trainer.read_text();after=patch.patch_dt_advantage_preprocessing(before)
                ast.parse(after);assert patch.patch_dt_advantage_preprocessing(after)==after
                assert after!=before and patch.RAY_ADV_INSERT_V8_PREVIOUS in before
                trainer.write_text(after,newline='\n')
                changes[str(trainer)]=dict(before=sha(old_verl/'verl/trainer/ppo/ray_trainer.py'),after=sha(trainer))
                for relative in ('verl/trainer/ppo/core_algos.py','verl/workers/fsdp_workers.py',
                        'verl/workers/actor/dp_actor.py','verl/utils/checkpoint/fsdp_checkpoint_manager.py',
                        'verl/utils/activation_offload.py','verl/utils/experimental/torch_functional.py',
                        'verl/utils/native_host_memory.py','verl/utils/logger/aggregate_logger.py'):
                    assert sha(verl/relative)==sha(old_verl/relative),relative
                env={k:(v.replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl)) if isinstance(v,str) else v)
                     for k,v in env.items()}
                env.update(DT_ENTRY_ROOT=str(entry),VERL_ROOT=str(verl),DT_CREDIT_MODE=MODE)
                bindings={path.replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl)):sha(
                    path.replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl))) for path in bindings}
                bindings.update({path:sha(path) for path in changes})
            record=dict(status='isolated_sources_prepared_not_verified_not_deployed',task=task,method=method,
                parent=str(parent),parent_SHA256=sha(parent),entry=str(entry),verl_root=str(verl),dt_root=row['dt_root'],
                effective_environment=env,source_bindings=bindings,changes=changes,
                credit_mode=MODE if method=='dt' else None,
                credit_variance_normalization='post_group_event_sum_no_mean_subtraction' if method=='dt' else 'unchanged',
                actor_gradient_norm='preclip_global_native_DTensor_full_tensor',local_shard_key='actor/grad_norm_local',
                extra_model_samples=0,extra_environment_samples=0)
            (AUDIT/f'{task}-{method}-source-prepared.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(task=task,method=method,changed_files=len(changes),entry=str(entry))),flush=True)

if __name__=='__main__':main()
