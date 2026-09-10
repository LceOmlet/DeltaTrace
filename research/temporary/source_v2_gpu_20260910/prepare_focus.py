"""Preserve completed shards and dispatch the user's five-task scope on this host."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/codex_source_v2_gpu_20260910_v1')
REPO = BASE / 'repo'
HERE = REPO / 'research/temporary/source_v2_gpu_20260910'
sys.path.insert(0, str(REPO / 'experiments/official'))
from summarize import summarize

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
old = BASE / 'full_v2'
output = BASE / 'focus_v1'
assert not output.exists()
release = json.loads((REPO / 'experiments/official/protocol.json').read_bytes())
state = json.loads((old / 'suite_status.json').read_bytes())
assert state['current_task'] == 'hotpotqa_long'
receipts = []
for task in ('vt_h4_c1', 'hotpotqa_long'):
    report = json.loads((old / task / 'results.json').read_bytes())
    assert report['status'] == 'complete'
    assert report['selected_counts'] == {task: release['tasks'][task]['count']}
    for name, field in [('evaluate.py', 'driver_sha256'), ('evidence_protocol.py', 'evidence_protocol_sha256')]:
        assert report[field] == state['plan']['code'][name] == sha(REPO / 'experiments/official' / name)
    assert sha(old / task / 'vectors.npz') == report['vectors_sha256']
    summary = summarize(report, release)
    (old / task / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    receipts.append({'dataset': task, 'results_sha256': sha(old / task / 'results.json'),
                     'vectors_sha256': report['vectors_sha256'], 'cases': len(report['cases'])})
gpu = subprocess.check_output(['/usr/bin/mx-smi', '--show-process'], text=True, timeout=30)
assert 'no process found' in gpu, gpu
pid = 5212
command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert str(HERE / 'run_suite.py') in command
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
for _ in range(50):
    if not Path(f'/proc/{pid}').exists():
        break
    time.sleep(.1)
assert not Path(f'/proc/{pid}').exists(), 'Old controller did not exit'
state.update(status='interrupted_for_user_focus', completed_tasks=receipts, current_task=None,
             cancellation='User canceled the six NI tasks; continue only four VT tasks and HotpotQA')
(old / 'suite_status.json').write_text(json.dumps(state, indent=2) + '\n')
output.mkdir()
for receipt in receipts:
    shutil.copytree(old / receipt['dataset'], output / receipt['dataset'])
    assert sha(output / receipt['dataset'] / 'results.json') == receipt['results_sha256']
tasks = ['vt_h4_c1', 'hotpotqa_long', 'vt_h2_c3', 'vt_h6_c1', 'vt_h10_c1']
command = [sys.executable, '-u', str(HERE / 'run_suite.py'), '--environment', str(BASE / 'environment.json'),
           '--output', str(output), '--stage', 'full', '--scope', 'vt_hotpot', '--task-order', *tasks]
with (BASE / 'focus_v1.log').open('w') as log:
    process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid': process.pid, 'command': command, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope': 'four VT tasks and HotpotQA', 'planned_cases': 448, 'reused_complete_tasks': receipts,
           'old_controller_exited': True, 'gpu_before': gpu, 'controller_sha256': sha(HERE / 'run_suite.py'),
           'focus_addendum_sha256': sha(HERE / 'FOCUS.md'), 'FT_K3_faithfulness_supplement': 'planned after primary paired run'}
(BASE / 'focus_execution_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'pid': process.pid, 'planned_cases': 448, 'reused_cases': 148}))
