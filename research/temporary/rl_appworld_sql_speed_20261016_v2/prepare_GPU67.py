"""Move already CPU-verified native trial configuration to user-freed GPU6/7."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD = ROOT / 'receipts/appworld-sql-native-speed-20261016-v1'
VERSION = 'appworld-sql-native-speed-20261016-v2'
AUDIT = ROOT / 'receipts' / VERSION
DEST = ROOT / 'candidates' / VERSION
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (AUDIT / 'prepared-index.json').exists(), 'Read existing preparation instead of replacing it'
stop = read(AUDIT / 'GPU67-user-stopped.json')
assert stop['status'] == 'stopped_at_user_request' and not stop['survivors']
index = read(OLD / 'prepared-index.json')
checks = read(OLD / 'CPU-verification.json')
assert len(checks) == 2 and all(c['status'] == 'original_Hydra_CPU_configuration_verified' for c in checks)
authorities = {str(ROOT / n): sha(ROOT / n) for n in ('formal-training.json', 'active-training.json', 'active-source.json')}
DEST.mkdir()
prepared = []
for item in index['tasks']:
    record = read(item['path'])
    assert all(sha(p) == h for p, h in record['source_bindings'].items())
    old_run = record['run']
    run = ROOT / 'runs' / VERSION / record['task'].lower()
    # Replace only existing output paths, leaving all numerical/task settings intact.
    options = record['options']
    options = {k: v.replace(old_run, str(run)) if isinstance(v, str) else v for k, v in options.items()}
    options['trainer.experiment_name'] = VERSION + '-' + record['task'].lower()
    env = dict(record['effective_environment'])
    env = {k: v.replace(old_run, str(run)) if isinstance(v, str) else v for k, v in env.items()}
    env['CUDA_VISIBLE_DEVICES'] = '6,7'
    env.pop('MACA_VISIBLE_DEVICES', None)
    for name in ('RAY_ADDRESS', 'RAY_TMPDIR'):
        env.pop(name, None)
    cpu_env = dict(env)
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    source = '''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
    argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', source, record['entry']],
                                             input=json.dumps(options), env=cpu_env, text=True))
    assert argv[0] == env['VENV_PYTHON'], 'Keep the original VENV_PYTHON path'
    record.update(version=VERSION, run=str(run), options=options, effective_environment=env,
                  argv=argv, physical_gpus=[6, 7], status='prepared_only',
                  previous_prepared=item['path'], previous_prepared_SHA256=sha(item['path']),
                  placement_change='User-authorized physical2/3 to6/7; same CPU-verified source/method/configuration')
    directory = DEST / record['task']
    directory.mkdir()
    path = directory / 'prepared.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    prepared.append(dict(task=record['task'], path=str(path)))
assert authorities == {p: sha(p) for p in authorities}
for name in ('CPU-verification.json', 'SQL-boundary-tests.xml'):
    shutil.copyfile(OLD / name, AUDIT / name)
reuse = dict(status='prior_CPU_checks_reused_not_rerun', source_version=index['version'],
             verified_configuration_changes=['CUDA_VISIBLE_DEVICES', 'independent output paths', 'experiment version label'],
             unchanged_source_checks=True, model_DT_optimizer_calls=0,
             receipts={str(OLD / n): sha(OLD / n) for n in ('CPU-verification.json', 'SQL-boundary-tests.xml')})
(AUDIT / 'CPU-verification-reuse.json').write_text(json.dumps(reuse, indent=2) + '\n')
(AUDIT / 'prepared-index.json').write_text(json.dumps(dict(version=VERSION, tasks=prepared,
    formal_authorities_unchanged=authorities, physical_gpus=[6, 7]), indent=2) + '\n')
print(json.dumps(dict(version=VERSION, tasks=prepared, physical_gpus=[6, 7], status='prepared_not_launched')))
