"""Reuse actual efficient native sources; prepare only, without training."""
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-reward-target-exponential-20261016-v5'
VERSION = 'textcraft-dt-reward-target-exponential-20261016-v5'
DEST = ROOT / 'candidates' / VERSION
RUN = ROOT / 'runs' / VERSION
OLD = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
WITHDRAWN = ROOT / 'runs/textcraft-dt-weighted-reward-targets-20261015-v4'
launch = json.loads((OLD / 'launch.json').read_bytes())
v4 = json.loads((WITHDRAWN / 'launch.json').read_bytes())
old_env = launch['effective_environment']
assert not DEST.exists(), 'Read the prepared receipt; never overwrite an existing candidate.'
DEST.mkdir()

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
original_files = json.loads((ROOT / 'receipts/textcraft-weighted-reward-targets-20261015-v4/baseline-manifest.json').read_bytes())
baseline = AUDIT / 'baseline'
baseline.mkdir(exist_ok=True)
for name, record in original_files.items():
    path = Path(record['path'])
    assert sha(path) == record['sha256'], (name, 'Actual baseline changed since its recorded receipt')
    shutil.copyfile(path, baseline / name)
for key, name in (('DT_ROOT', 'deltatrace'), ('VERL_ROOT', 'verl-owner')):
    shutil.copytree(old_env[key], DEST / name, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
entry = DEST / 'entry'
entry.mkdir()
for directory in old_env['PYTHONPATH'].split(':'):
    path = Path(directory) / 'owner_environment_configs.json'
    if path.is_file():
        shutil.copyfile(path, entry / path.name)
        break
else:
    raise FileNotFoundError('Native TextCraft owner environment configuration unavailable')
for path in (AUDIT / 'source/candidate-source').glob('*.py'):
    shutil.copyfile(path, entry / path.name)

env = dict(old_env)
env.update(DT_ROOT=str(DEST / 'deltatrace'), VERL_ROOT=str(DEST / 'verl-owner'),
           DT_ENTRY_ROOT=str(entry), DT_CREDIT_MODE='reward_target_exponential_v5',
           DT_FSDP_EVENT_WAIT_RECEIPTS=str(RUN / 'worker-runtime-overrides'))
env['PYTHONPATH'] = str(entry) + ':' + env['PYTHONPATH'].replace(old_env['DT_ROOT'], env['DT_ROOT']).replace(old_env['VERL_ROOT'], env['VERL_ROOT'])
for key in ('RAY_ADDRESS', 'RAY_TMPDIR', 'MACA_VISIBLE_DEVICES'):
    env.pop(key, None)
options = {key: value.replace(str(OLD), str(RUN)) if isinstance(value, str) else value
           for key, value in launch['options'].items()}
options['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = 'reward_target_exponential_v5'
options['trainer.experiment_name'] = VERSION
argv = [arg.replace(str(OLD), str(RUN)) for arg in launch['argv']]
argv = ['trainer.experiment_name='+json.dumps(VERSION) if arg.startswith('trainer.experiment_name=') else arg for arg in argv]
argv.append('+ray_init.runtime_env.env_vars.DT_CREDIT_MODE="reward_target_exponential_v5"')

bindings = {}
for name, record in original_files.items():
    original = Path(record['path'])
    if name in ('textcraft_owner_rollout.py', 'dt_training_batch.py', 'counterfactual.py'):
        current = entry / name
    elif name == 'native_ray_trainer.py':
        current = DEST / 'verl-owner/verl/trainer/ppo/ray_trainer.py'
    elif name == 'native_fsdp_workers.py':
        current = DEST / 'verl-owner/verl/workers/fsdp_workers.py'
    elif name.startswith('qwen35_'):
        current = DEST / 'deltatrace/clean/qwen35' / name
    else:
        current = original
    bindings[str(current)] = sha(current)
for path in entry.glob('*'):
    if path.is_file(): bindings[str(path)] = sha(path)

efficient = {
    'verl/utils/experimental/torch_functional.py': 'bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
    'verl/models/transformers/qwen3_vl.py': 'ebc52fb35812a9b0a5d9251f4e42e6fdbc15a076308390f2da81e47eea56a9d7',
    'verl/workers/actor/dp_actor.py': '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'verl/trainer/ppo/core_algos.py': 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'verl/utils/checkpoint/fsdp_checkpoint_manager.py': 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
}
for relative, expected in efficient.items():
    path = DEST / 'verl-owner' / relative
    assert sha(path) == expected, (relative, 'Wrong official/accepted efficiency owner')
    bindings[str(path)] = expected
for relative, expected in {
    'clean/qwen35/qwen35_answer_finite.py': '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e',
    'clean/qwen35/qwen35_dense_finite_runner.py': '198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd',
    'clean/qwen35/qwen35_gdn_finite.py': '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1',
    'accelerated/qwen35/qwen35_retained_capture.py': '5ad0bf9b8f9ec1868a20d964ad7a4a41c328f4da40c5d993f7ef92e8a5eaa810',
}.items():
    path = DEST / 'deltatrace' / relative
    assert sha(path) == expected, (relative, 'Wrong accepted numerical/memory DT owner')
    bindings[str(path)] = expected

fixed = ('actor_rollout_ref.model.lora_rank', 'actor_rollout_ref.model.lora_alpha',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu', 'actor_rollout_ref.actor.ppo_mini_batch_size',
    'actor_rollout_ref.actor.ppo_epochs', 'actor_rollout_ref.rollout.max_model_len',
    'actor_rollout_ref.rollout.enforce_eager', 'actor_rollout_ref.rollout.free_cache_engine',
    'trainer.save_freq', 'trainer.max_actor_ckpt_to_keep', 'trainer.total_epochs',
    'data.train_batch_size', 'env.rollout.n', 'trainer.resume_from_path', 'trainer.resume_mode')
assert all(options[key] == launch['options'][key] for key in fixed)
assert options['trainer.resume_from_path'] is None
record = dict(version=VERSION, method='dt', status='prepared_only_not_deployed', run=str(RUN),
    argv=argv, options=options, effective_environment=env, physical_gpus=[0,1], budget=launch['budget'],
    fresh_base_model=True, resume_checkpoint=None, upstream_commit=launch['upstream_commit'],
    parent_patch_commit=launch['local_patch_commit'], local_patch_commit=None,
    previous_launch=str(OLD/'launch.json'), previous_launch_SHA256=sha(OLD/'launch.json'),
    source_bindings=bindings, original_verified_files=original_files,
    variance_standardization=v4['variance_standardization'], PPO_reduction=v4['PPO_reduction'],
    method_changes=['Only native reward-settlement action target, including score 0.',
                    'Unscaled native EOS log effect; official g * -expm1(-d).',
                    'No affine seed, endpoint-difference or source-credit normalization.',
                    'Original official GRPO, variance-only scaling and PPO preserved.'],
    efficiency=dict(DT_numeric='26bef6c8', DT_gate_capture_host='3f51f5f7 GDN / original bounded replay',
                    native_vLLM_graph=True, frozen_head_patch='7a33e61a',
                    unaccepted_head_selection_enabled=False))
prepared = AUDIT / 'v5-prepared'
prepared.mkdir()
config = OmegaConf.load(OLD/'effective-config.yaml')
for key,value in options.items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
OmegaConf.save(config,prepared/'effective-config.yaml')
(prepared/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(AUDIT/'prepared-launch.json').write_bytes((prepared/'launch.json').read_bytes())
source=dict(version=VERSION,prepared_only=True,environment=env,
            upstream_commit=record['upstream_commit'],local_patch_commit=None,
            actual_inherited_sources=bindings,method_specification='experiments/rl/PLAN.md',
            parent_accepted_run=str(OLD),withdrawn_v4_not_used_for_DT_numeric=True)
(prepared/'source.json').write_text(json.dumps(source,indent=2)+'\n')
(AUDIT/'baseline-manifest.json').write_text(json.dumps(original_files,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],candidate=str(DEST),prepared=str(prepared),
                     fixed_configuration={key:options[key] for key in fixed},efficiency=record['efficiency'])))
