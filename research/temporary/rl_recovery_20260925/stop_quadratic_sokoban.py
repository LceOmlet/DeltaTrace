"""Stop only the manifested obsolete quadratic Sokoban run; preserve artifacts."""
import json
import os
from pathlib import Path
import signal
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
manifest = json.loads((root / 'formal-training.json').read_text())
job, = [j for j in manifest['jobs'] if j['task'] == 'Sokoban']
pid = job['pid']
assert pid == 3563544
assert str(Path(job['run_dir']) / 'run.sh') in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
assert os.getpgid(pid) == pid and os.getsid(pid) == pid
for other in manifest['jobs']:
    if other['task'] != 'Sokoban':
        assert os.getpgid(other['pid']) != pid
members = []
for path in Path('/proc').glob('[0-9]*'):
    try:
        if os.getpgid(int(path.name)) == pid:
            members.append(int(path.name))
    except ProcessLookupError:
        pass
assert 3615368 in members
out, = Path(job['ray_session']).glob('logs/worker-*-3615368.out')
tail = out.read_text(errors='replace').splitlines()[-8:]
receipt = dict(stopped_at=time.time(), reason='User forbids quadratic attribution calls; replace future-event cross product with one whole-return target per response',
               task='Sokoban', pid=pid, process_group=members, last_worker_lines=tail,
               checkpoint_dir=job['checkpoint_dir'], discarded='unfinished iteration only; files preserved')
target = root / 'receipts/rollout-major-cost/stop-quadratic-sokoban.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2)+'\n')
os.killpg(pid, signal.SIGTERM)
client = 735032
if Path(f'/proc/{client}/cmdline').exists():
    command = Path(f'/proc/{client}/cmdline').read_bytes().decode().replace('\0', ' ')
    assert 'apply_mixer_residency.py' in command and '--task Sokoban' in command
    os.kill(client, signal.SIGTERM)
for name in ['formal-training.json', 'active-training.json']:
    path = root / name
    data = json.loads(path.read_text())
    for item in data['jobs']:
        if item['task'] == 'Sokoban':
            item.update(status='stopped_for_linear_return_readout', stop_receipt=str(target), stopped_at=receipt['stopped_at'])
    path.write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps(receipt))
