"""Repair only the two reproduced native task-entry incompatibilities."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'appworld-sql-native-speed-20261016-v3'
OLD = ROOT / 'receipts/appworld-sql-native-speed-20261016-v2'
AUDIT = ROOT / 'receipts' / VERSION
DEST = ROOT / 'candidates' / VERSION
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(), 'Never overwrite a prior candidate or its receipts'
state = read(OLD / 'trial-status.json')
assert state['status'] == 'all_native_drivers_finished'
for item in state['tasks']:
    job = read(Path(item['run']) / 'job.json')
    assert job['exit_code'] == 1
AUDIT.mkdir(exist_ok=True)
DEST.mkdir()
authorities = {str(ROOT / n): sha(ROOT / n) for n in ('formal-training.json', 'active-training.json', 'active-source.json')}
prepared = []
checks = []
for item in read(OLD / 'prepared-index.json')['tasks']:
    record = read(item['path'])
    assert all(sha(p) == h for p, h in record['source_bindings'].items())
    previous_entry = record['entry']
    directory = DEST / record['task']
    entry = directory / 'entry'
    entry.mkdir(parents=True)
    for path in Path(previous_entry).iterdir():
        if path.is_file() and path.name in {Path(p).name for p in record['source_bindings'] if str(Path(p).parent) == previous_entry}:
            shutil.copyfile(path, entry / path.name)
    changes = []
    if record['task'] == 'SQL':
        path = entry / 'reward_readout.py'
        before = path.read_text()
        assert sha(path) == '1797faaaa89537f5369f82b18b613100f92344f2160f3ca68c5bddfb53009017'
        old = "if task not in ('TextCraft', 'AppWorld'):\n            raise ValueError('Direct action targets are enabled only for TextCraft and AppWorld')"
        new = "if task not in ('TextCraft', 'AppWorld', 'SkyRL-SQL'):\n            raise ValueError('Direct action targets require TextCraft, AppWorld or SkyRL-SQL native artifacts')"
        assert before.count(old) == 1
        after = before.replace(old, new)
        assert after.replace(new, old) == before
        path.write_text(after)
        changes.append(dict(path=str(path), old_SHA256=hashlib.sha256(before.encode()).hexdigest(), new_SHA256=sha(path),
                            scope='Task allowlist only; every numerical/mask/readout statement unchanged'))
    old_run = record['run']
    run = ROOT / 'runs' / VERSION / record['task'].lower()
    def placement(value):
        return value.replace(previous_entry, str(entry)).replace(old_run, str(run)) if isinstance(value, str) else value
    options = {k: placement(v) for k, v in record['options'].items()}
    options['trainer.experiment_name'] = VERSION + '-' + record['task'].lower()
    if record['task'] == 'AppWorld':
        assert options['actor_rollout_ref.rollout.mode'] == 'async'
        options['actor_rollout_ref.rollout.mode'] = 'sync'
        changes.append(dict(configuration='actor_rollout_ref.rollout.mode', before='async', after='sync',
                            scope='Existing native vLLM/VERL RPC interface; LOOP parallel task/cancellation owner unchanged'))
    env = {k: placement(v) for k, v in record['effective_environment'].items()}
    env['CUDA_VISIBLE_DEVICES'] = '6,7'
    env.pop('MACA_VISIBLE_DEVICES', None)
    cpu = dict(env)
    cpu['CUDA_VISIBLE_DEVICES'] = ''
    command = '''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
    argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', command, str(entry)],
                                             input=json.dumps(options), env=cpu, text=True))
    assert argv[0] == env['VENV_PYTHON']
    bindings = {p.replace(previous_entry, str(entry)): h for p, h in record['source_bindings'].items()}
    bindings.update({str(p): sha(p) for p in entry.iterdir() if p.is_file()})
    assert all(sha(p) == h for p, h in bindings.items())
    config = subprocess.run(argv + ['--cfg', 'job', '--resolve'], cwd=entry, env=cpu, capture_output=True)
    (directory / 'effective-config.yaml').write_bytes(config.stdout)
    (directory / 'CPU-config.stderr.log').write_bytes(config.stderr)
    config.check_returncode()
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(config.stdout.decode())
    assert cfg.actor_rollout_ref.rollout.mode == 'sync'
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.trainer.total_training_steps == 2
    if record['task'] == 'SQL':
        source = '''from types import SimpleNamespace
from reward_readout import DirectActionTargetReadout
for task in ('TextCraft','AppWorld','SkyRL-SQL'):
 r=DirectActionTargetReadout(SimpleNamespace(),SimpleNamespace(eos_token_id=1),task=task,packed_answer_targets=object())
 assert r.reward_target_exponential and not r.weighted_target_credit
 assert r.minibatch_size==4 and r.max_length==32768
print('Original constructor accepts three native task artifacts; v5 math flags preserved; no model calls')
'''
        check = subprocess.run([env['VENV_PYTHON'], '-c', source], cwd=entry, env=cpu, capture_output=True)
        (directory / 'CPU-task-registry.stdout.log').write_bytes(check.stdout)
        (directory / 'CPU-task-registry.stderr.log').write_bytes(check.stderr)
        check.check_returncode()
    record.update(version=VERSION, entry=str(entry), run=str(run), options=options, effective_environment=env,
                  argv=argv, source_bindings=bindings, status='prepared_only', previous_prepared=item['path'],
                  interface_fixes=changes, physical_gpus=[6, 7])
    path = directory / 'prepared.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    prepared.append(dict(task=record['task'], path=str(path)))
    checks.append(dict(task=record['task'], status='original_Hydra_CPU_configuration_verified',
                       config_SHA256=sha(directory / 'effective-config.yaml'), interface_fixes=changes,
                       source_bindings_verified=len(bindings), model_DT_optimizer_calls=0))
assert authorities == {p: sha(p) for p in authorities}
shutil.copyfile(OLD / 'SQL-boundary-tests.xml', AUDIT / 'SQL-boundary-tests.xml')
(AUDIT / 'CPU-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
(AUDIT / 'prepared-index.json').write_text(json.dumps(dict(version=VERSION, tasks=prepared,
    formal_authorities_unchanged=authorities, physical_gpus=[6, 7]), indent=2) + '\n')
print(json.dumps(dict(version=VERSION, prepared=prepared, checks=checks, status='prepared_not_launched')))
