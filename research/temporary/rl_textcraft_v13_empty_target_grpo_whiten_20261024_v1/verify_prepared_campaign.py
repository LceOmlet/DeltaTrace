"""Verify the actual native Hydra argv only; no worker or model execution."""
import json
import os
import subprocess
import time
import yaml
from prepare_campaign import AUDIT, RUN, VERSION, STATUS, sha, put


def main():
    assert not RUN.exists()
    row = json.loads((AUDIT / 'dt-prepared.json').read_text())
    environment = dict(os.environ, **row['effective_environment'])
    environment['CUDA_VISIBLE_DEVICES'] = ''
    resolved = subprocess.run(row['argv'] + ['--cfg', 'job'], cwd=row['verl_root'], env=environment,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240)
    (AUDIT / 'formal-CLI.stdout.yaml').write_text(resolved.stdout)
    (AUDIT / 'formal-CLI.stderr.log').write_text(resolved.stderr)
    assert resolved.returncode == 0, 'Inspect original Hydra stderr; no training has started'
    cfg = yaml.safe_load(resolved.stdout)
    assert cfg['trainer']['experiment_name'] == VERSION
    assert cfg['trainer']['total_epochs'] == 30 and cfg['trainer']['total_training_steps'] is None
    assert cfg['trainer']['n_gpus_per_node'] == 8
    assert cfg['trainer']['resume_mode'] == 'disable' and cfg['trainer']['resume_from_path'] is None
    assert cfg['actor_rollout_ref']['rollout']['engine_kwargs']['vllm']['mamba_ssm_cache_dtype'] == 'float32'
    assert cfg['data']['train_batch_size'] == 32 and cfg['env']['rollout']['n'] == 8
    assert cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 64
    assert cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] == 4
    assert cfg['ray_init']['runtime_env']['env_vars']['DT_CREDIT_MODE'] == 'reward_target_exponential_whiten_v13'
    assert cfg['ray_init']['runtime_env']['env_vars']['DT_NATIVE_HOST_CACHE_BATCH'] == '1'
    assert cfg['ray_init']['runtime_env']['env_vars']['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] == '0'
    assert all(sha(p) == h for p, h in row['source_bindings'].items())
    record = dict(status='passed_original_Hydra_parser_and_native_VERL_cfg_only', verified_unix=time.time(),
        native_argv=row['argv'], configuration_SHA256=sha(AUDIT / 'formal-CLI.stdout.yaml'),
        required_native_SSM_source_installation_pending=False, formal_training_started=False,
        model_calls=0, environment_calls=0, DT_calls=0, optimizer_updates=0)
    put(AUDIT / 'formal-CLI-verification.json', record)
    row.update(effective_config_SHA256=record['configuration_SHA256'],
               interface_verification=str(AUDIT / 'formal-CLI-verification.json'),
               interface_verification_SHA256=sha(AUDIT / 'formal-CLI-verification.json'))
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', dict(status=STATUS, order=['dt'], jobs=[row]))
    print(json.dumps(dict(status=record['status'], fresh_base_model=True, budget=row['budget'], formal_training_started=False)))


if __name__ == '__main__':
    main()
