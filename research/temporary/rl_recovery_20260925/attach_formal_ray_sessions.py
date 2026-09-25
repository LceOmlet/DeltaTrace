"""Record original driver/session identities after an authorized formal launch."""
import json
from pathlib import Path

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
training = json.loads((root / 'formal-training.json').read_text())
assert Path(training['dt_root']).is_dir()
drivers = {}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
        if b'-m verl.trainer.main_ppo' not in command:
            continue
        env = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
        directory = env.get(b'CHECKPOINT_DIR', b'').decode()
        if directory in [j['checkpoint_dir'] for j in training['jobs']]:
            assert env[b'DT_ROOT'].decode() == training['dt_root']
            assert directory not in drivers
            drivers[directory] = int(proc.name)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue

for job in training['jobs']:
    if job['checkpoint_dir'] not in drivers:
        continue
    driver = drivers[job['checkpoint_dir']]
    sessions = list((Path(job['ray_tmpdir']) / 'ray').glob('session_*_' + str(driver)))
    if len(sessions) == 1:
        job['driver_pid'] = driver
        job['ray_session'] = str(sessions[0])
        (Path(job['run_dir']) / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
for target in (root / 'formal-training.json', root / 'active-training.json',
               Path(training['jobs'][0]['run_dir']).parent / 'jobs.json'):
    target.write_text(json.dumps(training, indent=2) + '\n')
print(json.dumps({'all_sessions_recorded':all('ray_session' in j for j in training['jobs']),
    'jobs':[{k:j.get(k) for k in ('task','pid','driver_pid','ray_session','gpu')} for j in training['jobs']]}))
