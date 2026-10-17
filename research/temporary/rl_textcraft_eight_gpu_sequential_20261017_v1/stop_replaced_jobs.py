"""Stop only the identities replaced by the user's exclusive eight-GPU campaign.

Process termination is the unchanged block from the existing protected stop
owner. No checkpoint is created, altered, reconstructed, or restored here.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'
IDENTITIES = {(3823147, 1792200692.49), (3823185, 1792200693.05),
              (3823264, 1792200693.87), (2993563, 1792233768.81)}


def main(execute):
    manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
    roots, processes, seen = [], [], set()
    for pid, birth in sorted(IDENTITIES):
        try:
            p = psutil.Process(pid)
            assert p.create_time() == birth, 'PID now belongs to another process'
            if p.status() == psutil.STATUS_ZOMBIE:
                continue
        except psutil.NoSuchProcess:
            continue
        rows = [j for j in manifest['jobs'] if j.get('pid') == pid and j.get('birth') == birth]
        row = rows[0] if rows else None
        assert row is not None or pid == 2993563
        marker = None if row is None else Path(row['checkpoints']) / 'latest_checkpointed_iteration.txt'
        roots.append(dict(pid=pid, birth=birth, task=None if row is None else row['task'],
                          method=None if row is None else row['method'],
                          native_checkpoint_marker=None if marker is None or not marker.exists() else marker.read_text().strip()))
        for q in [p, *p.children(recursive=True)]:
            if q.pid not in seen:
                seen.add(q.pid)
                processes.append(q)
    record = dict(status='preview_not_stopped', collected=time.time(), roots=roots,
                  reason='User replaced concurrent experiments with exclusive eight-GPU fresh 200-iteration comparisons',
                  processes=[dict(pid=p.pid, birth=p.create_time(), name=p.name()) for p in processes],
                  new_checkpoints=0, training_state_writes=0, SIGSTOP_calls=0)
    if not execute:
        print(json.dumps(dict(status=record['status'], roots=roots, process_count=len(processes))))
        return
    AUDIT.mkdir(parents=True, exist_ok=True)
    final = AUDIT / 'user-replaced-prior-jobs-stop.json'
    assert not final.exists(), 'Inspect prior stop; do not repeat a completed transition'
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        (AUDIT / ('before-' + name)).write_bytes((ROOT / name).read_bytes())
    owner = AUDIT / 'source/resume_at_native_checkpoint.py'
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    tree = ast.parse(owner.read_text())
    template = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and 'processes=[parent]+parent.children(recursive=True)' in n.value)
    block = template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
    (AUDIT / 'user-replaced-prior-jobs-stopping.json').write_text(json.dumps(record, indent=2) + '\n')
    scope = dict(processes=processes, record=record, psutil=psutil, signal=signal, time=time)
    exec(compile(block, str(owner), 'exec'), scope)
    record.update(status='stopped_by_user_replacement', stop_owner=str(owner),
                  stop_owner_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest())
    final.write_text(json.dumps(record, indent=2) + '\n')
    assert not record['remaining_non_zombie'], 'Actual prior identities remain; do not launch eight-GPU work'
    for name in ('formal-training.json', 'active-training.json', 'active-source.json',
                 'receipts/appworld-formal-training-20261017-v1/formal-training.json'):
        path = ROOT / name
        value = json.loads(path.read_bytes())
        for job in value['jobs']:
            if (job.get('pid'), job.get('birth')) in IDENTITIES:
                job.update(status='stopped_by_user_for_fresh_eight_GPU_comparison', active=False, alive=False,
                           stopped_unix=record['finished_unix'], stop_receipt=str(final))
        temp = path.with_name(path.name + '.tmp-eight-gpu')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temp, path)
    print(json.dumps(dict(status=record['status'], roots=roots, stopped_processes=len(processes),
                          remaining=record['remaining_non_zombie'], receipt=str(final))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
