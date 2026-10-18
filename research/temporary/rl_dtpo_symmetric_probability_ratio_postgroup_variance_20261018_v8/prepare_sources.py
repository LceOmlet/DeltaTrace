"""Clone each actual method owner; DT method and common logging are explicit seams."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
MODE='reward_target_symmetric_probability_ratio_postgroup_variance_v8'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(AUDIT/'source'))
from patch_norm_logging import patch_actor,patch_formatter


def main():
    spec=importlib.util.spec_from_file_location('accepted_credit_patch',AUDIT/'source/patch_verl_agent2.py')
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            parent=ROOT/'receipts'/f'{task}-eight-gpu-symmetric-probability-ratio-20261018-v7/{method}-prepared.json'
            row=json.loads(parent.read_bytes());old_env=row['effective_environment']
            candidate=ROOT/'candidates'/f'{task}-{method}-symmetric-probability-ratio-postgroup-variance-20261018-v8'
            candidate.mkdir(exist_ok=False)
            old_entry=Path(row['entry']);old_verl=Path(row['verl_root'])
            entry=candidate/'entry';verl=candidate/'verl-owner'
            ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc')
            shutil.copytree(old_entry,entry,ignore=ignore);shutil.copytree(old_verl,verl,ignore=ignore)
            changes={}
            if method=='dt':
                for name in ('counterfactual.py','dt_training_batch.py','reward_readout.py','textcraft_owner_rollout.py'):
                    source=AUDIT/'source'/name
                    changes[str(entry/name)]={'before':sha(old_entry/name),'after':sha(source)}
                    shutil.copyfile(source,entry/name)
                trainer=verl/'verl/trainer/ppo/ray_trainer.py'
                after=patch.patch_dt_advantage_preprocessing(trainer.read_text())
                ast.parse(after);trainer.write_text(after,newline='\n')
                changes[str(trainer)]={'before':sha(old_verl/'verl/trainer/ppo/ray_trainer.py'),'after':sha(trainer)}
            for relative,transform in (('verl/workers/actor/dp_actor.py',patch_actor),
                    ('verl/utils/logger/aggregate_logger.py',patch_formatter)):
                target=verl/relative
                target.write_text(transform(target.read_text()),newline='\n')
                changes[str(target)]={'before':sha(old_verl/relative),'after':sha(target),'scope':'logging_only'}
            preserved={}
            for relative in ('verl/trainer/ppo/core_algos.py','verl/workers/fsdp_workers.py',
                    'verl/utils/fsdp_utils.py','verl/utils/checkpoint/fsdp_checkpoint_manager.py',
                    'verl/utils/activation_offload.py','verl/utils/experimental/torch_functional.py'):
                assert sha(verl/relative)==sha(old_verl/relative)
                preserved[relative]=sha(verl/relative)
            env={k:(v.replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl)) if isinstance(v,str) else v)
                 for k,v in old_env.items()}
            env.update(DT_ENTRY_ROOT=str(entry),VERL_ROOT=str(verl))
            if method=='dt':env['DT_CREDIT_MODE']=MODE
            bindings={}
            for p in row['source_bindings']:
                new=p.replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl))
                assert Path(new).is_file();bindings[new]=sha(new)
            bindings.update({p:sha(p) for p in changes})
            record=dict(status='isolated_sources_prepared_not_verified_not_deployed',task=task,method=method,
                parent=str(parent),parent_SHA256=sha(parent),entry=str(entry),verl_root=str(verl),
                dt_root=row['dt_root'],effective_environment=env,source_bindings=bindings,
                changes=changes,preserved_owners=preserved,credit_mode=MODE if method=='dt' else None,
                extra_model_calls=0,extra_environment_samples=0,
                credit_variance_normalization='post_group_event_sum_no_mean_subtraction' if method=='dt' else 'unchanged',
                actor_gradient_norm='preclip_global_native_DTensor_full_tensor',local_shard_key='actor/grad_norm_local')
            path=AUDIT/f'{task}-{method}-source-prepared.json'
            path.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(task=task,method=method,status=record['status'],candidate=str(candidate))))

if __name__=='__main__':main()
