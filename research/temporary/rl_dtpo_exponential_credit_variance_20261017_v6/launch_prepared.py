"""Submit verified native commands, preserving full native PPO restoration.

Only process ownership, provenance and formal manifests are managed here.
No model, algorithm, state reconstruction or checkpoint implementation exists
in this launcher.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil
from prepare_formal import ROOT, AUDIT, DEVICES, sha, write

CANONICAL_PPO = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json'


def launch(key):
    record = json.loads((AUDIT / (key + '-prepared.json')).read_bytes())
    assert record['status'] == 'CPU_configuration_and_behavior_verified_not_started'
    record['local_patch_commit'] = os.environ['RL_V6_SOURCE_COMMIT']
    assert re.fullmatch('[0-9a-f]{40}', record['local_patch_commit'])
    assert all(sha(p) == h for p, h in record['source_bindings'].items())
    task, method = record['task'], record['method']
    run = Path(record['run'])
    version = record['version']
    execution = (run / 'runtime-overrides' / version) if key == 'TextCraft_PPO' else run
    assert not (execution / 'launch.json').exists(), 'Inspect the existing identity; no duplicate submission'
    manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
    old, = [j for j in manifest['jobs'] if j.get('task') == task and j.get('method') == method]
    try:
        p = psutil.Process(old['pid'])
        assert p.create_time() != old['birth'] or p.status() == psutil.STATUS_ZOMBIE, 'Old formal driver still alive'
    except psutil.NoSuchProcess:
        pass
    protected = []
    for job in manifest['jobs']:
        if job is old:
            continue
        try:
            p = psutil.Process(job['pid'])
            if p.create_time() == job['birth'] and p.status() != psutil.STATUS_ZOMBIE:
                protected.append((p.pid, p.create_time()))
        except psutil.NoSuchProcess:
            pass
    physical = subprocess.check_output(['mx-smi'], text=True)
    processes = physical.split('| Process:')[-1]
    for device in record['physical_gpus']:
        assert not re.search(r'\|\s*' + str(device) + r'\s+\d+\s+', processes), 'Physical GPU occupied'
    observer = json.loads((ROOT / 'receipts/appworld-formal-training-20261017-v1/native-resource-observer-v1-job.json').read_bytes())
    p = psutil.Process(observer['pid'])
    assert p.create_time() == observer['birth']
    resources = json.loads((ROOT / 'receipts/appworld-formal-training-20261017-v1/latest-host-RSS-PSS-XTT-v1.json').read_bytes())
    assert time.time() - resources['collected'] < 60, 'Resource observation unavailable'
    # Preserve evidence; CPU preflight does not claim measured four-job safety.
    write(AUDIT / (key + '-resources-before-launch.json'), resources)
    (AUDIT / (key + '-physical-before-launch.txt')).write_text(physical)
    if key == 'TextCraft_PPO':
        saved = json.loads((ROOT / 'receipts/appworld-formal-training-20261017-v1/TextCraft-PPO-native-OOM-recovery-input-v2.json').read_bytes())
        assert saved['checkpoint'] == record['resume_checkpoint']
        assert sha(Path(saved['checkpoint']) / 'data.pt') == saved['data_SHA256']
        assert record['options']['trainer.resume_mode'] == 'resume_path'
        assert record['options']['trainer.resume_from_path'] == saved['checkpoint']
    execution.mkdir(parents=True, exist_ok=True)
    Path(record['output']).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(record['effective_environment'])
    for variable in ('MACA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'RAY_TMPDIR'):
        env.pop(variable, None)
    stdout = (run / ('driver.resume100-' + version + '.stdout.log')) if key == 'TextCraft_PPO' else Path(record['output']) / 'train.log'
    stderr = (run / ('driver.resume100-' + version + '.stderr.log')) if key == 'TextCraft_PPO' else execution / 'driver.stderr.log'
    # Same native Popen/argv/env boundary as the existing protected launchers.
    # Checkpoint state is loaded solely by original VERL trainer._load_checkpoint.
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(record['argv'], cwd=record['entry'], env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    birth = psutil.Process(process.pid).create_time()
    source_receipt = str(execution / 'source.json')
    record.update(pid=process.pid, birth=birth, observed_process_created_unix=birth,
        started_unix=time.time(), log=str(stdout), stdout=str(stdout), stderr=str(stderr),
        status='submitted_native_restore_pending' if key == 'TextCraft_PPO' else 'submitted_first_native_iteration_pending',
        devices=record['physical_gpus'], source_receipt=source_receipt,
        previous_job_identity=old, physical_four_job_safety_measured=False)
    write(execution / 'launch.json', record)
    write(execution / 'source.json', record)
    write(AUDIT / (key + '-launch-identity.json'), record)
    if key == 'TextCraft_PPO':
        prior = json.loads(CANONICAL_PPO.read_bytes())
        assert prior['pid'] == old['pid'] and prior['birth'] == old['birth']
        write(AUDIT / 'TextCraft-PPO-canonical-before-resource-recovery.json', prior)
        job = deepcopy(prior)
        job.update(pid=process.pid, birth=birth, observed_process_created_unix=birth,
            started_unix=record['started_unix'], status=record['status'], stdout=str(stdout), stderr=str(stderr),
            source_receipt=source_receipt, actual_execution_source_receipt=source_receipt,
            execution_version=version, execution_patch_commit=record['local_patch_commit'],
            effective_environment=record['effective_environment'], source_bindings=record['source_bindings'],
            resume_from_path=record['resume_checkpoint'], resumed_completed_iteration=100,
            previous_job_identity=prior, actual_native_restore_receipt=None,
            actual_native_binding_receipt=None, physical_gpus=[4, 5],
            resume_launch_receipt=str(execution/'launch.json'))
        write(CANONICAL_PPO, job)
    else:
        write(Path(record['output']) / 'job.json', record)
    fields = ('task', 'method', 'pid', 'birth', 'observed_process_created_unix', 'started_unix',
              'version', 'execution_version', 'status', 'run', 'output', 'log', 'stderr', 'checkpoints',
              'budget', 'entry', 'source_receipt', 'local_patch_commit', 'devices')
    row = {k: record[k] for k in fields}
    row.update(active=True, alive=True, verl_root=env['VERL_ROOT'], dt_root=env['DT_ROOT'],
               resume_from_path=record['resume_checkpoint'], resumed_completed_iteration=record['resumed_completed_iteration'],
               previous_job_identity=old, actual_native_restore_receipt=None)
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        path = ROOT / name
        current = json.loads(path.read_bytes())
        write(AUDIT / (key + '-before-' + name), current)
        current['jobs'] = [j for j in current['jobs'] if not (j.get('task') == task and j.get('method') == method)] + [row]
        current['four_formal_v6_resource_receipts'] = str(AUDIT)
        current['updated_unix'] = time.time()
        write(path, current)
    campaign = ROOT / 'receipts/appworld-formal-training-20261017-v1/formal-training.json'
    current = json.loads(campaign.read_bytes())
    current['jobs'] = [j for j in current['jobs'] if not (j.get('task') == task and j.get('method') == method)] + [row]
    write(campaign, current)
    for pid, created in protected:
        assert psutil.Process(pid).create_time() == created, 'A protected formal process changed identity'
    print(json.dumps({k: record[k] for k in ('key', 'version', 'pid', 'birth', 'status', 'physical_gpus',
                                             'resume_checkpoint', 'checkpoints')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key', choices=DEVICES)
    launch(parser.parse_args().key)
