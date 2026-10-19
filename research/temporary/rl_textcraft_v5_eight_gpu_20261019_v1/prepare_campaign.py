"""Select the existing v5 mode on the accepted eight-GPU sources and workload."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1'
VERSION = 'textcraft-dtpo-v5-exponential-8gpu-200-20261019-v1'
MODE = 'reward_target_exponential_v5'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    from omegaconf import OmegaConf
    prior = ROOT/'receipts/textcraft-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8/dt-prepared.json'
    row = json.loads(prior.read_bytes())
    before = copy.deepcopy(row)
    for p, expected in row['source_bindings'].items():
        assert sha(p) == expected, (p, 'actual provisioned source differs from accepted binding')
    run = ROOT/'runs'/VERSION
    checkpoint = ROOT.parent.parent/'deltatrace_backups/textcraft-8gpu-20261017'/VERSION/'checkpoints'
    assert not run.exists() and not checkpoint.exists()
    old_run = row['run']
    def rebind(value):
        if isinstance(value, str):
            return value.replace(old_run,str(run)).replace(before['version'],VERSION).replace(
                'reward_target_symmetric_probability_ratio_postgroup_variance_v8', MODE)
        return value
    row['options'] = {k:rebind(v) for k,v in row['options'].items()}
    row['effective_environment'] = {k:rebind(v) for k,v in row['effective_environment'].items()}
    row['effective_environment']['DT_CREDIT_MODE'] = MODE
    row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = MODE
    row['options']['trainer.default_local_dir'] = str(checkpoint)
    row['options']['trainer.experiment_name'] = VERSION
    row['argv'] = [rebind(arg) for arg in before['argv']]
    for key,value in row['options'].items():
        if value != before['options'].get(key):
            prefix=key+'='
            assert sum(arg.startswith(prefix) for arg in row['argv'])==1,key
            row['argv']=[prefix+json.dumps(value) if arg.startswith(prefix) else arg for arg in row['argv']]
    row.update(version=VERSION,run=str(run),output=str(run/'textcraft-dt'),checkpoints=str(checkpoint),
        status='prepared_only_not_submitted',method_version='DTPO-v5-original-exponential',
        credit_variance_normalization='original_v5_post_group_advantage_variance_no_mean_subtraction',
        method_source_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
        execution_parent_commit=before['local_patch_commit'],
        parent_source_receipt=str(prior),parent_source_SHA256=sha(prior),
        source_commit=os.environ.get('RL_CAMPAIGN_COMMIT'),
        fresh_base_model=True,resume_checkpoint=None,prior_tuning_preparation=None,
        local_patch_commit=os.environ.get('RL_CAMPAIGN_COMMIT'),
        method_specification='experiments/rl/PLAN.md',
        actor_gradient_norm='preclip_global_native_DTensor_full_tensor',local_shard_key='actor/grad_norm_local')
    changed={k:dict(before=before['options'].get(k),after=v) for k,v in row['options'].items() if before['options'].get(k)!=v}
    allowed={'trainer.default_local_dir','trainer.experiment_name','trainer.rollout_data_dir','trainer.validation_data_dir',
        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS','+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'}
    assert set(changed)<=allowed, changed
    assert row['options']['trainer.resume_mode']=='disable' and row['options']['trainer.resume_from_path'] is None
    assert row['options']['trainer.total_training_steps']==200 and row['options']['trainer.n_gpus_per_node']==8
    assert row['options']['data.train_batch_size']==32 and row['options']['env.rollout.n']==8
    assert row['options']['actor_rollout_ref.actor.ppo_mini_batch_size']==64
    assert row['options']['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
    assert row['options']['actor_rollout_ref.rollout.max_model_len']==32768
    assert row['options']['actor_rollout_ref.actor.ppo_epochs']==1
    assert row['options']['algorithm.norm_adv_by_std_in_grpo'] is True
    assert row['options']['trainer.save_freq']==5
    assert row['options']['trainer.max_actor_ckpt_to_keep']==2
    config_path = prior.parent/'dt-effective-config.yaml'
    config = OmegaConf.load(config_path)
    for key,value in row['options'].items():
        OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    OmegaConf.save(config,AUDIT/'dt-effective-config.yaml')
    row['effective_config_SHA256'] = sha(AUDIT/'dt-effective-config.yaml')
    (AUDIT/'dt-prepared.json').write_text(json.dumps(row,indent=2)+'\n')
    (AUDIT/'campaign-prepared.json').write_text(json.dumps(dict(
        status='native_v5_eight_gpu_configuration_resolved',order=['dt'],jobs=[row],
        configuration_changes=changed,training_algorithm_code_changes=0),indent=2)+'\n')
    cleanup=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
    assert sha(cleanup)=='b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    shutil.copyfile(cleanup,AUDIT/'source/resume_at_native_checkpoint.py')
    original_tests=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source'
    (AUDIT/'baseline').mkdir(exist_ok=True)
    shutil.copyfile(original_tests.parent/'baseline/counterfactual.py',AUDIT/'baseline/counterfactual.py')
    for name in ('test_behavior_v5.py','test_multi_target_v5.py'):
        text=(original_tests/name).read_text()
        text=text.replace("AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
            "AUDIT=ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1'")
        (AUDIT/'source'/name).write_text(text)
    print(json.dumps(dict(status='native_v5_eight_gpu_configuration_resolved',version=VERSION,
        configuration_changes=changed,algorithm_code_changes=0,prepared=str(AUDIT/'dt-prepared.json'))))

if __name__ == '__main__':
    main()
