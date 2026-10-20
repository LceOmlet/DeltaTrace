"""Select the existing v5 calculation on the current verified native sources."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v5-credit-integration-20261020-v1'
VERSION = 'textcraft-dtpo-v5-exponential-verified-8gpu-200-20261020-v1'
MODE = 'reward_target_exponential_v5'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from omegaconf import OmegaConf
    prior = ROOT / 'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4/dt-prepared.json'
    before = json.loads(prior.read_bytes())
    row = copy.deepcopy(before)
    for path, digest in before['source_bindings'].items():
        assert sha(path) == digest, (path, 'recorded provisioned source differs')
    run = ROOT / 'runs' / VERSION
    checkpoint = ROOT.parent.parent / 'deltatrace_backups/textcraft-8gpu-20261017' / VERSION / 'checkpoints'
    assert not run.exists() and not checkpoint.exists()
    old_run, old_version = before['run'], before['version']

    def rebind(value):
        if not isinstance(value, str):
            return value
        return value.replace(old_run, str(run)).replace(old_version, VERSION).replace(
            'reward_target_exponential_clipped_v9', MODE)

    row['options'] = {k: rebind(v) for k, v in before['options'].items()}
    row['effective_environment'] = {k: rebind(v) for k, v in before['effective_environment'].items()}
    env = row['effective_environment']
    env['DT_CREDIT_MODE'] = MODE
    env.pop('DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP', None)
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(run / 'worker-runtime-overrides')
    row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = MODE
    row['options'].pop('+ray_init.runtime_env.env_vars.DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP', None)
    row['options']['+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'] = env['DT_FSDP_EVENT_WAIT_RECEIPTS']
    row['options']['trainer.default_local_dir'] = str(checkpoint)
    row['options']['trainer.experiment_name'] = VERSION
    row['argv'] = [rebind(a) for a in before['argv'] if not a.startswith('+ray_init.runtime_env.env_vars.DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP=')]
    for key, value in row['options'].items():
        if value != before['options'].get(key):
            prefix = key + '='
            assert sum(a.startswith(prefix) for a in row['argv']) == 1, key
            row['argv'] = [prefix + json.dumps(value) if a.startswith(prefix) else a for a in row['argv']]
    row.update(version=VERSION, run=str(run), output=str(run/'textcraft-dt'), checkpoints=str(checkpoint),
        status='prepared_only_not_submitted', method_version='DTPO-v5-original-unclipped-exponential',
        credit_variance_normalization='original_v5_post_group_advantage_variance_no_mean_subtraction',
        method_source_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
        execution_parent_commit=before['local_patch_commit'], parent_source_receipt=str(prior),
        parent_source_SHA256=sha(prior), source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'], fresh_base_model=True,
        resume_checkpoint=None, prior_tuning_preparation=None, method_specification='experiments/rl/PLAN.md')
    changed = {k: {'before': before['options'].get(k), 'after': v} for k, v in row['options'].items() if v != before['options'].get(k)}
    allowed = {'trainer.default_local_dir', 'trainer.experiment_name', 'trainer.rollout_data_dir',
        'trainer.validation_data_dir', '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS',
        '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'}
    assert set(changed) <= allowed, changed
    assert row['options']['trainer.resume_mode'] == 'disable'
    assert row['options']['trainer.resume_from_path'] is None
    for key, expected in {'trainer.total_training_steps': 200, 'trainer.n_gpus_per_node': 8,
        'data.train_batch_size': 32, 'env.rollout.n': 8,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 64,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.max_model_len': 32768,
        'actor_rollout_ref.actor.ppo_epochs': 1, 'trainer.save_freq': 5,
        'trainer.max_actor_ckpt_to_keep': 2}.items():
        assert row['options'][key] == expected, key
    assert row['options']['algorithm.norm_adv_by_std_in_grpo'] is True
    config = OmegaConf.load(prior.parent/'dt-effective-config.yaml')
    config.ray_init.runtime_env.env_vars.pop('DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP', None)
    for key, value in row['options'].items():
        OmegaConf.update(config, key.lstrip('+'), value, force_add=True)
    OmegaConf.save(config, AUDIT/'dt-effective-config.yaml')
    row['effective_config_SHA256'] = sha(AUDIT/'dt-effective-config.yaml')
    (AUDIT/'dt-prepared.json').write_text(json.dumps(row, indent=2)+'\n')
    (AUDIT/'campaign-prepared.json').write_text(json.dumps(dict(
        status='native_v5_eight_gpu_configuration_resolved', order=['dt'], jobs=[row],
        configuration_changes=changed, training_algorithm_code_changes=0), indent=2)+'\n')
    cleanup = ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
    assert sha(cleanup) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    shutil.copyfile(cleanup, AUDIT/'source/resume_at_native_checkpoint.py')
    old = ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
    (AUDIT/'baseline').mkdir(exist_ok=True)
    shutil.copyfile(old/'baseline/counterfactual.py', AUDIT/'baseline/counterfactual.py')
    shutil.copyfile(old/'native-multi-target-artifacts.pt', AUDIT/'native-multi-target-artifacts.pt')
    for name in ['test_behavior_v5.py', 'test_multi_target_v5.py']:
        code = (old/'source'/name).read_text().replace(
            "AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
            "AUDIT=ROOT/'receipts/textcraft-v5-credit-integration-20261020-v1'")
        (AUDIT/'source'/name).write_text(code)
    print(json.dumps({'status': 'prepared_only', 'version': VERSION,
        'configuration_changes': changed, 'algorithm_code_changes': 0}))


if __name__ == '__main__':
    main()
