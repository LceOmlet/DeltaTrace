"""Configure three fresh eight-GPU native VERL jobs; no training implementation.

The same accepted v6 task adapter and fixed actor/core serve all three methods.
Only the official advantage selector and PPO's native critic distinguish them.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'textcraft-eight-gpu-sequential-20261017-v1'
AUDIT = ROOT / 'receipts' / VERSION
CANDIDATE = ROOT / 'candidates' / VERSION
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main(task='TextCraft'):
    global VERSION, AUDIT, CANDIDATE
    assert task in ('TextCraft', 'AppWorld')
    VERSION = task.lower() + '-eight-gpu-sequential-20261017-v1'
    AUDIT = ROOT / 'receipts' / VERSION
    CANDIDATE = ROOT / 'candidates' / VERSION
    appworld = task == 'AppWorld'
    assert not CANDIDATE.exists(), 'Preserve existing preparation; inspect rather than replace'
    manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
    selected = {j['method']: j for j in manifest['jobs'] if j.get('task') == task
                and j.get('method') in ('dt', 'ppo')}
    sources = {k: json.loads(Path(j['source_receipt']).read_bytes()) for k, j in selected.items()}
    D, P = sources['dt'], sources['ppo']
    base_env = D['effective_environment']
    original_entry, original_verl = Path(base_env['DT_ENTRY_ROOT']), Path(base_env['VERL_ROOT'])
    resource_dt = ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace'
    assert sha(resource_dt / 'clean/qwen35/qwen35_dense_finite_runner.py') == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    CANDIDATE.mkdir(parents=True)
    entry, verl = CANDIDATE / 'entry', CANDIDATE / 'verl-owner'
    shutil.copytree(original_entry, entry, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(original_verl, verl, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    # Already deployed and naturally executed native critic-return resource seam.
    # Its arithmetic is unchanged, as are the actor, core, checkpoint and DT math.
    worker = Path(P['effective_environment']['VERL_ROOT']) / 'verl/workers/fsdp_workers.py'
    worker_sha = ('12a74ca90182384b2a1deddcb4179f14b620bfd85b6471ac0ccf0836dc1a3b29' if appworld else
                  '198f509c9a0af8a03d54f22e66f5f0d5593ab4c9eb7b27777c201ef323d2fecb')
    assert sha(worker) == worker_sha
    shutil.copyfile(worker, verl / 'verl/workers/fsdp_workers.py')
    if appworld:
        # Preserve the accepted AppWorld native critic's 32k memory backend;
        # the v6 trainer and entry still come from the DTPO v6 parent.
        for name, digest in {
            'verl/workers/critic/dp_critic.py': '5aadf696b1e9b11057b13cef6a08a6f56f56aabeb7d3e5ddd63ada11d7fd9694',
            'verl/utils/activation_offload.py': '242f7919ef00bcf4dced4d4c856946d2e761dbf24297d434356194f03943cfcb',
        }.items():
            source = Path(P['effective_environment']['VERL_ROOT']) / name
            assert sha(source) == digest
            shutil.copyfile(source, verl / name)
    preserved = {
        'verl/workers/actor/dp_actor.py': '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
        'verl/trainer/ppo/core_algos.py': 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
        'verl/utils/checkpoint/fsdp_checkpoint_manager.py': 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
        'verl/utils/activation_offload.py': ('242f7919ef00bcf4dced4d4c856946d2e761dbf24297d434356194f03943cfcb' if appworld else
                                          '829042503f87abeeff2e4c86b27b409acb04d71b503820bbb5fa80eb7bd1d950'),
        'verl/utils/experimental/torch_functional.py': 'bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
        'verl/models/transformers/qwen3_vl.py': 'ebc52fb35812a9b0a5d9251f4e42e6fdbc15a076308390f2da81e47eea56a9d7',
    }
    for name, digest in preserved.items():
        assert sha(verl / name) == digest, name
    bindings = {str(p): sha(p) for p in entry.glob('*') if p.is_file()}
    for name in [*preserved, 'verl/workers/fsdp_workers.py', 'verl/trainer/ppo/ray_trainer.py',
                 'verl/trainer/main_ppo.py', 'verl/utils/native_host_memory.py', 'verl/workers/critic/dp_critic.py']:
        bindings[str(verl / name)] = sha(verl / name)
    if appworld:
        assert sha(verl / 'verl/trainer/ppo/ray_trainer.py') == 'b7c8d65ad236fcf0c622ae81bec4916a62169deed8e2e4277bca8af1ac58ea0b'
        # The actual task sampler, environment and checkpoint sidecar remain
        # in their already provisioned, pinned LOOP owner.
        for path in D['source_bindings']:
            if '/loop-owner/' in path:
                bindings[path] = sha(path)
    for p in resource_dt.rglob('*.py'):
        original = Path(base_env['DT_ROOT']) / p.relative_to(resource_dt)
        if p.name != 'qwen35_dense_finite_runner.py':
            assert p.read_bytes() == original.read_bytes(), str(p)
    bindings[str(resource_dt / 'clean/qwen35/qwen35_dense_finite_runner.py')] = sha(resource_dt / 'clean/qwen35/qwen35_dense_finite_runner.py')
    configs, rows = {}, []
    for method in ('dt', 'ppo', 'grpo'):
        parent = P if method == 'ppo' else D
        parent_job = selected['ppo' if method == 'ppo' else 'dt']
        tag = 'dtpo-v6' if method == 'dt' else method
        version = task.lower() + '-' + tag + '-8gpu-200-20261017-v1'
        run = ROOT / 'runs' / version
        output = run / (task.lower() + '-' + method)
        checkpoint = Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups') / (task.lower() + '-8gpu-20261017') / version / 'checkpoints'
        assert not run.exists() and not checkpoint.exists(), 'Fresh run must not reuse earlier state'
        env = deepcopy(base_env)
        mapping = [(str(original_entry), str(entry)), (str(original_verl), str(verl)),
                   (base_env['DT_ROOT'], str(resource_dt))]
        for before, after in mapping:
            env['PYTHONPATH'] = env['PYTHONPATH'].replace(before, after)
        env.update(DT_ENTRY_ROOT=str(entry), VERL_ROOT=str(verl), DT_ROOT=str(resource_dt),
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', DT_NATIVE_HOST_CACHE_REPLAY_LAYER='1',
                   DT_FSDP_EVENT_WAIT_RECEIPTS=str(run / 'worker-runtime-overrides'))
        for key in ('DT_NATIVE_HOST_MEMORY_LEASE', 'MACA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'RAY_TMPDIR'):
            env.pop(key, None)
        if method != 'dt':
            env.pop('DT_CREDIT_MODE', None)
        options = deepcopy(parent['options'])
        if appworld:
            options['data.custom_cls.path'] = str(entry / 'loop_iteration_dataset.py')
        for key in list(options):
            if key.endswith('.DT_NATIVE_HOST_MEMORY_LEASE') or key.endswith('.DT_CREDIT_MODE'):
                options.pop(key)
        options.update({'algorithm.adv_estimator': {'dt': 'deltatrace', 'ppo': 'gae', 'grpo': 'grpo'}[method],
            'trainer.n_gpus_per_node': 8, 'trainer.total_training_steps': 200, 'trainer.total_epochs': 200 if appworld else 30,
            'trainer.save_freq': 2 if appworld else 5, 'trainer.max_actor_ckpt_to_keep': 2,
            'trainer.experiment_name': version, 'trainer.default_local_dir': str(checkpoint),
            'trainer.rollout_data_dir': str(output / 'rollouts'),
            'trainer.validation_data_dir': str(output / 'validation'),
            'trainer.resume_mode': 'disable', 'trainer.resume_from_path': None})
        if method in ('dt', 'grpo'):
            options['algorithm.norm_adv_by_std_in_grpo'] = True
        for key in ('DT_NATIVE_HOST_CACHE_BATCH', 'DT_NATIVE_HOST_CACHE_REPLAY_LAYER', 'DT_CREDIT_MODE', 'DT_FSDP_EVENT_WAIT_RECEIPTS'):
            if key in env:
                options['+ray_init.runtime_env.env_vars.' + key] = env[key]
        # The original argv constructor and Hydra resolver own configuration.
        cpu_env = dict(os.environ, **env)
        cpu_env['CUDA_VISIBLE_DEVICES'] = ''
        cpu_env.pop('DT_NATIVE_HOST_MEMORY_LEASE', None)
        command = ('import json,sys; from owner_runtime_options import owner_command; '
                   'print(json.dumps(owner_command(json.loads(sys.stdin.read()))))')
        argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', command],
                         input=json.dumps(options), env=cpu_env, text=True))
        assert argv[0] == env['VENV_PYTHON']
        resolved = subprocess.run(argv + ['--cfg', 'job', '--resolve'], cwd=entry, env=cpu_env, capture_output=True)
        (AUDIT / (method + '-effective-config.yaml')).write_bytes(resolved.stdout)
        (AUDIT / (method + '-CPU-config.stderr')).write_bytes(resolved.stderr)
        resolved.check_returncode()
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(resolved.stdout.decode())
        assert cfg.trainer.n_gpus_per_node == 8 and cfg.trainer.total_training_steps == 200
        assert cfg.data.train_batch_size == (40 if appworld else 32) and cfg.env.rollout.n == (6 if appworld else 8)
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == (32 if appworld else 64)
        assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
        assert cfg.actor_rollout_ref.actor.ppo_epochs == (2 if appworld else 1)
        assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
        assert cfg.trainer.save_freq == (2 if appworld else 5) and cfg.trainer.max_actor_ckpt_to_keep == 2
        if appworld:
            assert cfg.trainer.test_freq == 5 and cfg.trainer.val_before_train is False
            assert cfg.data.loop_train_groups == 40 and cfg.data.loop_eval_groups == 57
            if method == 'ppo':
                assert cfg.critic.ppo_micro_batch_size_per_gpu == 4
                assert cfg.critic.model.activation_offload_native_max_seq_len == 8192
        assert 'DT_NATIVE_HOST_MEMORY_LEASE' not in cfg.ray_init.runtime_env.env_vars
        assert cfg.actor_rollout_ref.rollout.enforce_eager is False
        configs[method] = OmegaConf.to_container(cfg, resolve=True)
        row = dict(task=task, method=method, version=version, run=str(run), output=str(output),
            checkpoints=str(checkpoint), entry=str(entry), verl_root=str(verl), dt_root=str(resource_dt),
            argv=argv, options=options, effective_environment=env, source_bindings=bindings,
            parent_source_receipt=parent_job['source_receipt'],
            parent_source_SHA256=sha(parent_job['source_receipt']),
            fresh_base_model=True, resume_checkpoint=None, status='prepared_not_started',
            devices=list(range(8)), physical_gpus=list(range(8)),
            budget=dict(total_iterations=200, groups=40 if appworld else 32,
                        trajectories_per_group=6 if appworld else 8,
                        optimizer_minibatch=32 if appworld else 64,
                        ppo_epochs=2 if appworld else 1,
                        optimizer_updates_per_iteration='native retained trajectories and partial minibatches' if appworld else 4),
            paper_budget=dict(iterations=200), local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'],
            upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
            effective_config_SHA256=sha(AUDIT / (method + '-effective-config.yaml')),
            verified_DT_resource_receipt=str(ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3/execution2/native-comparison-summary-v1.json'))
        if appworld:
            assert parent['sampler_checkpoint_artifact'] == 'loop_sampler.pt'
            row['sampler_checkpoint_artifact'] = parent['sampler_checkpoint_artifact']
            row['accepted_sampler_checkpoint_source'] = parent['sampler_checkpoint_extension']
        write(AUDIT / (method + '-prepared.json'), row)
        rows.append(row)
    # Fairness checks compare actual owner-resolved configuration, not a second
    # implementation of loss scaling, sampling, optimization or seed behavior.
    equal_sections = ['data', 'env', 'actor_rollout_ref']
    common = {name: configs['dt'][name] for name in equal_sections}
    for method in ('ppo', 'grpo'):
        for name in equal_sections:
            assert configs[method][name] == common[name], (method, name)
    receipt = dict(status='three_native_CPU_configs_resolved_and_common_workload_equal',
        collected=time.time(), version=VERSION, jobs=rows, order=['dt', 'ppo', 'grpo'],
        common_sections=equal_sections, common_effective_configuration=common,
        unchanged_numerical_owners=preserved, new_training_calls=0,
        pending_scope='Confirmed TextCraft and AppWorld; three algorithms per task, six fresh runs in sequence',
        native_actor_global_microbatches_per_full_minibatch=8 if appworld else 16,
        native_actor_DP_ranks=8, native_actor_microbatches_per_rank_per_full_minibatch=1 if appworld else 2,
        source_bindings=bindings, local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'])
    write(AUDIT / 'campaign-prepared.json', receipt)
    print(json.dumps(dict(status=receipt['status'], order=receipt['order'], source_bindings=len(bindings),
                         task=task, devices=list(range(8)), iterations_per_combination=200)))


if __name__ == '__main__':
    main()
