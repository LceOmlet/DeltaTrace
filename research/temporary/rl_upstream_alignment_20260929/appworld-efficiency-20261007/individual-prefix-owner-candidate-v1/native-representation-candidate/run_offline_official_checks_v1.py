"""Sequential original-owner checks on sealed real B8 diagnostic operands.

This entry is only staged by its preparer. Explicit invocation runs no Ray,
model, optimizer or checkpoint operation. Any failed owner check saves its
unaltered stdout/stderr and stops; no tolerance or numeric correction exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rank0', action='store_true')
    group.add_argument('--rank1', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    prepared_path = directory / 'prepared.json'
    prepared = json.loads(prepared_path.read_bytes())
    ranks = [0] if args.rank0 else [1] if args.rank1 else [0, 1]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Explicitly use CUDA_VISIBLE_DEVICES=2 for this isolated check.')
    if os.environ.get('FLA_CI_ENV') == '1':
        raise ValueError('Do not enable the original FLA CI warning-only exception.')
    # Same physical process-table occupancy expression as the existing B8
    # launcher. No Torch import or device initialization occurs before it.
    physical = subprocess.check_output(['mx-smi'], text=True)
    if re.search(r'^\|\s+2\s+\d+\s+', physical.split('| Process:')[-1], re.M):
        raise RuntimeError('Physical GPU2 is occupied; leave all other processes unchanged.')
    for item in prepared['sources'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Sealed source changed: ' + item['path'])
    if sha(prepared['parent_result']['path']) != prepared['parent_result']['sha256']:
        raise ValueError('Completed parent result changed.')
    # Identify actual operands from each sealed final rank record. No filename
    # heuristic or old run/CP data is used as a fallback.
    records = {}
    for rank in ranks:
        item = prepared['rank_records'][str(rank)]
        if sha(item['path']) != item['sha256']:
            raise ValueError('Final parent rank record changed.')
        record = json.loads(Path(item['path']).read_bytes())
        if record['phase'] != 'native_prefix_lease_diagnostic_complete':
            raise ValueError('Parent rank did not finish the original bounded diagnostic.')
        for key in ('native_gdn0', 'native_fa3', 'actual_finite'):
            value = record[key]
            if sha(value['path']) != value['sha256']:
                raise ValueError('Actual saved operands changed: ' + value['path'])
        records[rank] = record

    import psutil
    run = directory / ('execution-' + '-'.join(map(str, ranks)) + '-' + str(time.time_ns()))
    run.mkdir()
    receipt = dict(scope=__doc__, pid=os.getpid(), pid_birth=psutil.Process().create_time(),
        started_unix=time.time(), physical_devices=[2], ranks=ranks,
        entry=dict(path=__file__, sha256=sha(__file__)),
        prepared=dict(path=str(prepared_path), sha256=sha(prepared_path)),
        parent_result=prepared['parent_result'], checks=[], status='running',
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        model=False, Ray=False, optimizer=False, checkpoint=False)
    receipt_path = run / 'execution.json'

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')

    save()
    print(json.dumps(dict(execution=str(receipt_path), pid=receipt['pid'],
                         pid_birth=receipt['pid_birth'], ranks=ranks)), flush=True)
    sources = prepared['sources']
    try:
        for rank in ranks:
            record = records[rank]
            target = run / f'rank{rank}'
            target.mkdir()
            commands = [
                ('gdn0', [sys.executable, '-u', sources['gdn_observer']['path'],
                    '--operands', record['native_gdn0']['path'], '--official-source', sources['FLA_tests']['path'],
                    '--output', str(target / 'original-gdn0.json'), '--device', 'cuda']),
                ('varlen', [sys.executable, '-u', sources['varlen_observer']['path'],
                    '--operands', record['native_fa3']['path'], '--official-source', sources['FA_tests']['path'],
                    '--existing-checker', sources['dense_observer']['path'],
                    '--output', str(target / 'original-varlen-output.json'), '--device', 'cuda']),
                ('finite', [sys.executable, '-u', sources['actual_row_helper']['path'],
                    '--native-operands', record['native_fa3']['path'],
                    '--finite-operands', record['actual_finite']['path'],
                    '--saved-verifier', sources['saved_verifier']['path'],
                    '--sources', str(Path(sources['FA_tests']['path']).parent),
                    '--row-wrapper', sources['row_wrapper']['path'], '--row-library', sources['row_library']['path'],
                    '--scalar-wrapper', sources['scalar_wrapper']['path'], '--scalar-library', sources['scalar_library']['path'],
                    '--environment-json', sources['numerical_environment']['path'],
                    '--output-dir', str(target / 'original-finite-rows')]),
            ]
            for name, command in commands:
                stdout, stderr = target / (name + '.stdout.txt'), target / (name + '.stderr.txt')
                check = dict(rank=rank, name=name, command=command, started_unix=time.time(),
                             stdout=str(stdout), stderr=str(stderr), status='running')
                receipt['checks'].append(check)
                save()
                with stdout.open('xb') as out, stderr.open('xb') as err:
                    child = subprocess.Popen(command, cwd=directory, stdout=out, stderr=err)
                    check.update(pid=child.pid, pid_birth=psutil.Process(child.pid).create_time())
                    save()
                    code = child.wait()
                check.update(returncode=code, finished_unix=time.time(),
                    status='passed' if code == 0 else 'failed',
                    stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
                save()
                if code:
                    raise RuntimeError(f'Original {name} check failed on rank{rank}; unaltered stderr: {stderr}')
        receipt['status'] = 'passed'
    except BaseException:
        receipt['status'] = 'failed'
        path = run / 'failure-traceback.txt'
        path.write_text(traceback.format_exc(), encoding='utf8')
        receipt['failure_traceback'] = dict(path=str(path), sha256=sha(path))
        raise
    finally:
        receipt['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
