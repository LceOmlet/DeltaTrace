"""Resolve existing native commands for v6 and the two unchanged PPO controls.

This module manages configuration/provenance only. VERL owns training, state
restoration, checkpointing and the original stopping budgets.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
MODE = 'reward_target_exponential_variance_v6'
DEVICES = {'TextCraft_DTPO': [0, 1], 'AppWorld_DTPO': [6, 7],
           'TextCraft_PPO': [4, 5], 'AppWorld_PPO': [2, 3]}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(key):
    source = json.loads((AUDIT / (key + '-source-prepared.json')).read_bytes())
    candidate = ROOT / 'candidates' / source['version']
    destination = candidate / 'prepared.json'
    assert not destination.exists(), 'Retain previous preparation and execution identity'
    assert source['prepared_resource_revision'] == 2
    assert all(sha(p) == h for p, h in source['source_bindings'].items())
    old = json.loads(Path(source['parent_launch']).read_bytes())
    old_env = old['effective_environment']
    dt = key.endswith('DTPO')
    app = key.startswith('AppWorld')
    task = 'AppWorld' if app else 'TextCraft'
    method = 'dt' if dt else 'ppo'
    version = source['version']
    run = ROOT / 'runs' / version if dt or app else Path(old['run'])
    # Preserve the established output leaf convention used by the native reader.
    output = run / (('appworld-' if app else 'textcraft-') + method)
    old_run = Path(source['parent_run'])
    old_entry = old_env.get('DT_ENTRY_ROOT', old_env['PYTHONPATH'].split(':')[0])
    checkpoints = (Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/appworld-20261017')
                   / version / 'checkpoints') if app else output / 'checkpoints'
    mapping = [(old_entry, source['entry']), (old_env['VERL_ROOT'], source['verl_root'])]
    if dt or app:
        mapping += [(str(old_run), str(run)),
                    (old['options']['trainer.default_local_dir'], str(checkpoints))]

    def translate(value):
        if isinstance(value, str):
            for before, after in mapping:
                value = value.replace(before, after)
            return value
        if isinstance(value, dict):
            return {k: translate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [translate(v) for v in value]
        return value

    options = translate(deepcopy(old['options']))
    env = translate(deepcopy(source['effective_environment']))
    # Numerical DT source is the actually accepted parent; it is not renamed.
    assert env['DT_ROOT'] == old_env['DT_ROOT']
    for variable in ('MACA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'RAY_TMPDIR'):
        env.pop(variable, None)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, DEVICES[key]))
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(run / 'worker-runtime-overrides')
    if dt:
        env['DT_CREDIT_MODE'] = MODE
    for variable in ('DT_CREDIT_MODE', 'DT_NATIVE_HOST_MEMORY_LEASE',
                     'DT_NATIVE_HOST_CACHE_BATCH', 'DT_FSDP_EVENT_WAIT_RECEIPTS'):
        if variable in env:
            options['+ray_init.runtime_env.env_vars.' + variable] = env[variable]
    checkpoint = None
    resumed_step = 0
    if dt or app:
        options.update({'trainer.experiment_name': version,
                        'trainer.default_local_dir': str(checkpoints),
                        'trainer.rollout_data_dir': str(output / 'rollouts'),
                        'trainer.validation_data_dir': str(output / 'validation'),
                        'trainer.resume_mode': 'disable', 'trainer.resume_from_path': None})
    else:
        saved = json.loads((ROOT / 'receipts/appworld-formal-training-20261017-v1'
                            / 'TextCraft-PPO-native-OOM-recovery-input-v2.json').read_bytes())
        checkpoint, resumed_step = saved['checkpoint'], saved['completed_iteration']
        assert resumed_step == 100
        assert int((Path(checkpoint).parent / 'latest_checkpointed_iteration.txt').read_text()) == resumed_step
        assert sha(Path(checkpoint) / 'data.pt') == saved['data_SHA256']
        options.update({'trainer.resume_mode': 'resume_path', 'trainer.resume_from_path': checkpoint})
    expected = {
        'verl/workers/actor/dp_actor.py': '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
        'verl/trainer/ppo/core_algos.py': 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
        'verl/utils/checkpoint/fsdp_checkpoint_manager.py': 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
        'verl/utils/experimental/torch_functional.py': 'bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
        'verl/models/transformers/qwen3_vl.py': 'ebc52fb35812a9b0a5d9251f4e42e6fdbc15a076308390f2da81e47eea56a9d7',
    }
    expected['verl/utils/activation_offload.py'] = (
        '242f7919ef00bcf4dced4d4c856946d2e761dbf24297d434356194f03943cfcb' if app and not dt
        else '829042503f87abeeff2e4c86b27b409acb04d71b503820bbb5fa80eb7bd1d950')
    expected['verl/workers/fsdp_workers.py'] = (
        '5bb0211a5cf5bd355bb763da93d26e3902f1297c96273eddf5e3a6640d455dd3' if app and not dt
        else '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3')
    for relative, digest in expected.items():
        assert sha(Path(source['verl_root']) / relative) == digest, relative
    for relative, digest in {
        'clean/qwen35/qwen35_gdn_finite.py': '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1',
        'clean/qwen35/qwen35_dense_finite_runner.py': '198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd',
        'clean/qwen35/qwen35_answer_finite.py': '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e',
    }.items() if dt else ():
        assert sha(Path(env['DT_ROOT']) / relative) == digest, relative
    cpu = dict(os.environ, **env)
    cpu['CUDA_VISIBLE_DEVICES'] = ''
    cpu.pop('DT_NATIVE_HOST_MEMORY_LEASE', None)
    code = ('import json,sys; sys.path.insert(0,sys.argv[1]); '
            'from owner_runtime_options import owner_command; '
            'print(json.dumps(owner_command(json.loads(sys.stdin.read()))))')
    argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', code, source['entry']],
                     input=json.dumps(options), env=cpu, text=True))
    assert argv[0] == env['VENV_PYTHON']
    resolved = subprocess.run(argv + ['--cfg', 'job', '--resolve'], cwd=source['entry'], env=cpu,
                              capture_output=True)
    (AUDIT / (key + '-effective-config.yaml')).write_bytes(resolved.stdout)
    (AUDIT / (key + '-CPU-config.stderr.log')).write_bytes(resolved.stderr)
    resolved.check_returncode()
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(resolved.stdout.decode())
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.trainer.save_freq == (2 if app else 5)
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.algorithm.adv_estimator == ('deltatrace' if dt else 'gae')
    if dt:
        assert cfg.algorithm.norm_adv_by_std_in_grpo is True
        assert cfg.ray_init.runtime_env.env_vars.DT_CREDIT_MODE == MODE
    if app:
        assert cfg.trainer.total_training_steps == cfg.trainer.total_epochs == 200
        assert cfg.data.train_batch_size == 40 and cfg.env.rollout.n == 6
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 32
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
        assert cfg.trainer.default_local_dir == str(checkpoints)
    else:
        assert cfg.trainer.total_epochs == 30 and cfg.trainer.total_training_steps is None
        assert cfg.data.train_batch_size == 32 and cfg.env.rollout.n == 8
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    checkpoints.mkdir(parents=True, exist_ok=True)
    # The original full native checkpoint remains the sole restore input.
    changes = {k: {'before': old['options'].get(k), 'after': v}
               for k, v in options.items() if old['options'].get(k) != v}
    record = dict(source, task=task, method=method, execution_version=version,
        run=str(run), output=str(output), checkpoints=str(checkpoints), argv=argv, options=options,
        effective_environment=env, physical_gpus=DEVICES[key], budget=old['budget'],
        fresh_base_model=bool(dt or app), resume_checkpoint=checkpoint,
        resumed_completed_iteration=resumed_step, configuration_changes=changes,
        status='CPU_configuration_and_behavior_verified_not_started',
        effective_config_SHA256=sha(AUDIT / (key + '-effective-config.yaml')),
        CPU_behavior_receipt=str(AUDIT / 'CPU-v6-and-v5-default.xml'),
        sampler_checkpoint_artifact=old.get('sampler_checkpoint_artifact'),
        sampler_checkpoint_extension=old.get('sampler_checkpoint_extension'),
        accepted_PPO_receipt=old.get('accepted_PPO_receipt'),
        parent_method_source_commit=old.get('method_source_commit', old.get('local_patch_commit')),
        local_patch_commit=os.environ.get('RL_V6_SOURCE_COMMIT', 'pending_frozen_commit_not_deployed'),
        upstream_commits=old.get('upstream_commits', {'VERL': '20bd331bdbc9026a5668e11362178e10ab7400c8'}),
        original_official_GRPO_normalized=bool(cfg.algorithm.norm_adv_by_std_in_grpo) if dt else None,
        resource_patch_numerical_body_AST_verified=True,
        cache_release_or_queue_latency_is_not_claimed_free=True,
        prepared_unix=time.time())
    write(destination, record)
    write(AUDIT / (key + '-prepared.json'), record)
    print(json.dumps({k: record[k] for k in ('key', 'version', 'status', 'run', 'physical_gpus',
                                           'checkpoints', 'resume_checkpoint', 'resumed_completed_iteration')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key', choices=DEVICES)
    prepare(parser.parse_args().key)
