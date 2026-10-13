"""Withdraw only the premature DT candidate; preserve all evidence and GRPO."""
import json
import os
from pathlib import Path
import signal
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT / 'runs/textcraft-dt-global-token-mean-temporal-20261014-v1'
receipt = OUT / 'premature-temporal-candidate-stop-v1.json'
if receipt.exists():
    prior = json.loads(receipt.read_bytes())
    assert prior.get('status') == 'stopped_pending_analytical_norm_accounting', prior
    print(json.dumps({'receipt': str(receipt), 'report': prior}))
    raise SystemExit(0)

pid, birth = 2510247, 1791907554.02
grpo_pid, grpo_birth = 3174448, 1791793629.73
protected = psutil.Process(grpo_pid)
assert protected.create_time() == grpo_birth
protected_pids = {grpo_pid, *[p.pid for p in protected.children(recursive=True)]}
driver = psutil.Process(pid)
assert driver.create_time() == birth
assert os.getsid(pid) == pid == os.getpgid(pid)
members = []
for p in psutil.process_iter():
    try:
        if os.getpgid(p.pid) == pid:
            assert os.getsid(p.pid) == pid and p.pid not in protected_pids
            members.append({'pid': p.pid, 'birth': p.create_time(), 'name': p.name()})
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass

report = {
    'started_unix': time.time(), 'pid': pid, 'birth': birth,
    'reason': 'Premature causal hypothesis withdrawn: no prior analytical evidence that B4 denominator bias explains accelerating temporal gradient norms.',
    'status': 'validated_identity_before_stop', 'members_before': members,
    'formal_grpo_preserved': {'pid': grpo_pid, 'birth': grpo_birth},
    'signals': [], 'files_preserved': True, 'new_checkpoint_created': False,
    'formal_manifests_changed': False, 'training_parameters_changed': False,
}
receipt.write_text(json.dumps(report, indent=2) + '\n')
os.killpg(pid, signal.SIGTERM)
report['signals'].append({'signal': 'SIGTERM', 'scope': 'validated isolated process group', 'unix': time.time()})

def live_members():
    live = []
    for m in members:
        try:
            p = psutil.Process(m['pid'])
            if p.create_time() == m['birth'] and p.status() != psutil.STATUS_ZOMBIE:
                assert os.getsid(p.pid) == pid and os.getpgid(p.pid) == pid
                assert p.pid not in protected_pids
                live.append(p)
        except psutil.NoSuchProcess:
            pass
    return live

deadline = time.monotonic() + 12
while live_members() and time.monotonic() < deadline:
    time.sleep(.25)
for p in live_members():
    observed_birth = p.create_time()
    p.kill()
    report['signals'].append({'signal': 'SIGKILL', 'pid': p.pid, 'birth': observed_birth, 'unix': time.time()})
time.sleep(1)
report['live_survivors'] = [p.pid for p in live_members()]
assert not report['live_survivors']
protected = psutil.Process(grpo_pid)
assert protected.create_time() == grpo_birth and protected.status() != psutil.STATUS_ZOMBIE
report['formal_grpo_preserved']['alive_after'] = True
report['status'] = 'stopped_pending_analytical_norm_accounting'
report['finished_unix'] = time.time()
for subdir in ('temporal-update-records', 'runtime-overrides'):
    report[subdir] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted((RUN/subdir).rglob('*')) if p.is_file()]
marker = RUN/'textcraft-dt/checkpoints/latest_checkpointed_iteration.txt'
report['checkpoint_marker'] = {'path': str(marker), 'value': marker.read_text().strip() if marker.exists() else None}
launch = RUN/'launch.json'
before = OUT/'temporal-candidate-launch-before-withdrawal-v1.json'
assert not before.exists()
before.write_bytes(launch.read_bytes())
data = json.loads(launch.read_bytes())
data['status'] = report['status']
data['withdrawal_receipt'] = str(receipt)
data['withdrawal_reason'] = report['reason']
launch.write_text(json.dumps(data, indent=2) + '\n')
receipt.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'receipt': str(receipt), 'report': report}))
