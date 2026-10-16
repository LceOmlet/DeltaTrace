"""Compose a bounded native PPO run from the verified LOOP and PPO owners."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'appworld-native-ppo-speed-20261016-v1'
AUDIT = ROOT / 'receipts' / VERSION
DEST = ROOT / 'candidates' / VERSION
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if DEST.exists():
    assert not (DEST / 'prepared.json').exists() and not (ROOT / 'runs' / VERSION).exists()
    # Preserve the completed first CPU composition whose source-receipt scan
    # subsequently used a nonexistent critic.yaml path. No model was launched.
    for name in ('effective-config.yaml', 'CPU-config.stderr.log'):
        previous = DEST / name
        if previous.exists() and not previous.with_suffix(previous.suffix + '.metadata-attempt1').exists():
            shutil.copyfile(previous, previous.with_suffix(previous.suffix + '.metadata-attempt1'))
    (AUDIT / 'CPU-source-receipt-attempt1-failed.json').write_text(json.dumps(dict(
        error='Nonexistent verl/trainer/config/critic.yaml in extra source-binding scan',
        original_Hydra_composition_completed=True, model_DT_optimizer_calls=0), indent=2) + '\n')
base_path = ROOT / 'candidates/appworld-sql-native-speed-20261016-v3/AppWorld/prepared.json'
base = read(base_path)
ppo_path = ROOT / 'runs/textcraft-ppo-control-fixed-verl-fresh-20261014-v2/launch.json'
ppo = read(ppo_path)
assert all(sha(path) == expected for path, expected in base['source_bindings'].items())
assert ppo['options']['algorithm.adv_estimator'] == 'gae'
assert base['options']['algorithm.adv_estimator'] == 'deltatrace'
authorities = {str(ROOT / name): sha(ROOT / name) for name in ('formal-training.json', 'active-training.json', 'active-source.json')}
DEST.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)
entry = base['entry']
run = ROOT / 'runs' / VERSION / 'appworld'
options = dict(base['options'])
for key, value in list(options.items()):
    if isinstance(value, str):
        options[key] = value.replace(base['run'], str(run))
options['algorithm.adv_estimator'] = 'gae'
for key, value in ppo['options'].items():
    if key.startswith('critic.'):
        options[key] = value
options['trainer.experiment_name'] = VERSION
env = dict(os.environ)
env.update(base['effective_environment'])
env['CUDA_VISIBLE_DEVICES'] = '2,3'
env.pop('MACA_VISIBLE_DEVICES', None)
for key, value in list(env.items()):
    if isinstance(value, str):
        env[key] = value.replace(base['run'], str(run))
cpu = dict(env)
cpu['CUDA_VISIBLE_DEVICES'] = ''
command = '''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', command, entry],
                                         input=json.dumps(options), env=cpu, text=True))
assert argv[0] == env['VENV_PYTHON']
config = subprocess.run(argv + ['--cfg', 'job', '--resolve'], env=cpu, cwd=entry, capture_output=True)
(DEST / 'effective-config.yaml').write_bytes(config.stdout)
(DEST / 'CPU-config.stderr.log').write_bytes(config.stderr)
config.check_returncode()
from omegaconf import OmegaConf
cfg = OmegaConf.create(config.stdout.decode())
assert cfg.algorithm.adv_estimator == 'gae'
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
assert cfg.critic.ppo_micro_batch_size_per_gpu == 4
assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 32
assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
assert cfg.trainer.total_training_steps == 2
assert cfg.trainer.save_freq == -1 and cfg.trainer.test_freq == -1
bindings = dict(base['source_bindings'])
for suffix in ('verl/workers/critic/dp_critic.py',):
    path = Path(env['VERL_ROOT']) / suffix
    bindings[str(path)] = sha(path)
changes = {key:dict(before=base['options'].get(key), after=value) for key, value in options.items()
           if base['options'].get(key) != value}
record = dict(base, version=VERSION, run=str(run), argv=argv, options=options,
              effective_environment=env, physical_gpus=[2,3], status='prepared_only',
              source_bindings=bindings, previous_prepared=str(base_path),
              algorithm='Original fixed VERL GAE/PPO and native critic; no DT calls',
              native_critic_template=dict(path=str(ppo_path), SHA256=sha(ppo_path)),
              changes_from_DT_trial=changes)
path = DEST / 'prepared.json'
path.write_text(json.dumps(record, indent=2) + '\n')
assert authorities == {path:sha(path) for path in authorities}
(AUDIT / 'prepared-index.json').write_text(json.dumps(dict(version=VERSION,
    tasks=[dict(task='AppWorld', path=str(path))], formal_authorities_unchanged=authorities), indent=2) + '\n')
checks = dict(status='original_Hydra_GAE_PPO_configuration_verified', config_SHA256=sha(DEST / 'effective-config.yaml'),
              model_DT_optimizer_calls=0, changes=changes,
              critic_resolved=OmegaConf.to_container(cfg.critic, resolve=True))
(AUDIT / 'CPU-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
shutil.copyfile(ROOT / 'receipts/appworld-sql-native-speed-20261016-v3/SQL-boundary-tests.xml', AUDIT / 'SQL-boundary-tests.xml')
print(json.dumps(dict(version=VERSION, prepared=str(path), changes=changes,
    actor_epochs=cfg.actor_rollout_ref.actor.ppo_epochs, critic_epochs=cfg.critic.ppo_epochs,
    critic_minibatch=cfg.critic.ppo_mini_batch_size, source_bindings=len(bindings))))
