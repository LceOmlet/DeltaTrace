"""Thin fresh process/provenance boundary for the two prepared native jobs.

Run on MetaX after sourcing the recorded original metax-entry.env.sh. Default
mode only checks the frozen manifests and prints their launch identities.
--execute is intentionally separate and is used only after owner acceptance.
The unchanged prepared argv/environment perform all training and task work.
No stop, checkpoint load/export, model call, service call or backup is added.

Transport staging continues to use stage_environment_entry.SSH/SCP. The
standard Popen/PID-birth/manifest pattern follows submit_prepared_fresh.py;
that older one-shot script cannot be called because it hardcodes retired PID
and service identities. This file does not claim to be an upstream trainer.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil


TASKS = {'TextCraft': ('textcraft', [2, 3]), 'AppWorld': ('appworld', [4, 5])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def checked(item):
    assert sha(item['path']) == item['sha256'], item['path']
    return read(item['path'])


def check_prepared(candidate, task):
    slug, devices = TASKS[task]
    prepared_path = candidate/slug/'prepared.json'
    prep = read(prepared_path)
    assert prep['status'] == 'prepared_CPU_imports_passed_not_submitted', prep['status']
    plan, source = checked(prep['launch_plan']), checked(prep['source_template'])
    imports = checked(prep['CPU_imports'])
    env, options = plan['environment'], source['startup_options']
    assert imports['options'] == options
    assert prep['task'] == plan['task'] == task
    assert prep['devices'] == plan['devices'] == devices
    if 'run_environment' in prep:
        assert checked(prep['run_environment']) == env
    assert source['environment'] == env
    assert plan['entry'] == env['DT_ENTRY_ROOT'] == source['entry']
    assert env['VERL_ROOT'] == source['verl_root'] and env['DT_ROOT'] == source['dt_root']
    assert env['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, devices))
    assert plan['resume_mode'] == source['resume_mode'] == options['trainer.resume_mode'] == 'disable'
    assert not plan['checkpoint_restore_requested'] and not source['checkpoint_restore_requested']
    assert 'trainer.resume_from_path' not in options
    assert '--resume-from' not in plan['argv'] and '--config-only' not in plan['argv']
    assert plan['argv'][0] == env['VENV_PYTHON']
    assert plan['argv'][1] == str(Path(env['DT_ENTRY_ROOT'])/f'launch_{slug}_native.py')
    assert str(plan['working_directory']) in plan['argv']
    assert env['DT_MAX_LENGTH'] == '32768'
    for key, value in {
        'actor_rollout_ref.model.lora_rank': 8,
        'actor_rollout_ref.model.lora_alpha': 16,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 4,
    }.items():
        assert options[key] == value, key
    # Read the committed source identity from the frozen manifests. Never
    # substitute this script's local HEAD or a historical directory name.
    commit = source['local_patch_commit']
    assert re.fullmatch(r'[0-9a-f]{40}', commit), commit
    if 'repository_commit' in prep:
        assert prep['repository_commit'] == commit
    bindings = source['source_bindings']
    assert bindings and str(Path(plan['argv'][1])) in bindings
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    return dict(task=task, devices=devices, prepared_path=str(prepared_path),
                prepared_sha256=sha(prepared_path), prep=prep, plan=plan, source=source,
                committed_source=commit, source_bindings_verified=len(bindings))


def check_previous_owner(active, task):
    old = next((job for job in active['jobs'] if job['task'] == task), None)
    if old is None:
        return None
    try:
        previous = psutil.Process(old['pid'])
        same = previous.create_time() == old['observed_process_created_unix']
        assert not (same and previous.status() != psutil.STATUS_ZOMBIE), (
            task, 'Previous identified driver is live; no stop is performed')
    except psutil.NoSuchProcess:
        pass
    # Reuse the prior submit owner's orphan guard. It identifies only the
    # old output command; it neither signals nor adopts another process.
    for process in psutil.process_iter(['pid', 'cmdline', 'status']):
        if process.info['status'] == psutil.STATUS_ZOMBIE:
            continue
        assert not any(str(old['output']) in argument
                       for argument in (process.info['cmdline'] or [])), (
                           task, 'Previous output command is still live', process.info['pid'])
    return old


def check_devices(devices):
    physical = subprocess.check_output(['mx-smi'], text=True)
    processes = physical.split('| Process:')[-1]
    for device in devices:
        assert not re.search(r'^\|\s+'+str(device)+r'\s+\d+\s+', processes, re.M), (
            'Requested physical GPU is occupied', device)
    return physical


def submit_one(root, item, receipt_dir, physical):
    task, plan, source = item['task'], item['plan'], copy.deepcopy(item['source'])
    active, sources = read(root/'active-training.json'), read(root/'active-source.json')
    old = check_previous_owner(active, task)
    output = Path(plan['working_directory'])
    assert not output.exists(), 'Inspect the existing submission instead of repeating it'
    output.mkdir(parents=True)
    for name in ('active-training.json', 'active-source.json', 'formal-training.json'):
        (output/('prior-'+name)).write_bytes((root/name).read_bytes())
    (output/'physical-before.txt').write_text(physical)
    source.update(unix=time.time(), prepared_only=False,
                  prepared_receipt=item['prepared_path'],
                  prepared_receipt_sha256=item['prepared_sha256'],
                  submission_script_sha256=sha(__file__),
                  runtime_verification_status='submitted; actual worker imports/update not yet observed',
                  checkpoint_restore_requested=False, resume_mode='disable')
    for binding in [dict(path=item['prepared_path'], sha256=item['prepared_sha256']),
                    item['prep']['launch_plan'], item['prep']['source_template'],
                    dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))]:
        source['source_bindings'][binding['path']] = binding['sha256']
    write(output/'source.json', source)
    # Preserve every prepared argument and environment value verbatim.
    # Platform runtime variables come from the original sourced environment.
    environment = dict(os.environ, **plan['environment'])
    with (output/'train.log').open('wb') as log:
        child = subprocess.Popen(plan['argv'], env=environment, cwd=output,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job = dict(task=task, method='dt', pid=child.pid,
               observed_process_created_unix=psutil.Process(child.pid).create_time(),
               started_unix=time.time(), devices=item['devices'], entry=source['entry'],
               verl_root=source['verl_root'], dt_root=source['dt_root'], argv=plan['argv'],
               output=str(output), log=str(output/'train.log'), checkpoints=str(output/'checkpoints'),
               source_receipt=str(output/'source.json'), status='fresh_formal_submitted_not_yet_verified',
               lora_rank=8, lora_alpha=16, actor_microbatch=4,
               log_prob_micro_batch_size_per_gpu=4, budget=plan['budget'],
               checkpoint_restore_requested=False, resume_mode='disable')
    if 'LOOP_ROOT' in environment and task == 'AppWorld':
        job['loop_root'] = environment['LOOP_ROOT']
    write(output/'job.json', job)
    jobs = [job if prior['task'] == task else prior for prior in active['jobs']]
    if old is None:
        jobs.append(job)
    manifest = dict(active, manifest=str(output.parent/'formal-training.json'), jobs=jobs)
    if old is not None:
        manifest.setdefault('retired_jobs', []).append(dict(
            old, replacement_pid=child.pid, status='replaced_by_fresh_base_run_without_checkpoint'))
    for path in (output.parent/'formal-training.json', root/'active-training.json', root/'formal-training.json'):
        write(path, manifest)
    source_job = dict(task=task, pid=job['pid'], entry=job['entry'], verl_root=job['verl_root'],
                      dt_root=job['dt_root'], source_receipt=job['source_receipt'], status=job['status'],
                      runtime_override=None, actor_microbatch=4, lora_rank=8, lora_alpha=16)
    existed = any(prior['task'] == task for prior in sources['jobs'])
    sources['jobs'] = [source_job if prior['task'] == task else prior for prior in sources['jobs']]
    if not existed:
        sources['jobs'].append(source_job)
    sources.update(unix=time.time(), manifest=manifest['manifest'])
    write(root/'active-source.json', sources)
    receipt = dict(job=job, source_sha256=sha(output/'source.json'),
                   prepared=dict(path=item['prepared_path'], sha256=item['prepared_sha256']),
                   committed_source=item['committed_source'],
                   launch_plan=item['prep']['launch_plan'],
                   submission_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                   source_bindings_verified=item['source_bindings_verified'],
                   scope='Fresh standard Popen only; unchanged prepared native argv/environment; no checkpoint operations or training-health claim')
    write(receipt_dir/(TASKS[task][0]+'-submission.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--receipt-dir', type=Path)
    args = parser.parse_args()
    if args.execute and args.receipt_dir is None:
        parser.error('--execute requires a new --receipt-dir')
    assert len(args.tasks) == len(set(args.tasks))
    items = [check_prepared(args.candidate_root, task) for task in args.tasks]
    if not args.execute:
        print(json.dumps(dict(status='prepared_only_no_process_or_manifest_change',
                              tasks=[dict(task=item['task'], devices=item['devices'],
                                          prepared_path=item['prepared_path'],
                                          prepared_sha256=item['prepared_sha256'],
                                          committed_source=item['committed_source'],
                                          argv=item['plan']['argv'],
                                          source_bindings_verified=item['source_bindings_verified'])
                                     for item in items]), indent=2))
        return
    assert not args.receipt_dir.exists(), 'Inspect the existing submission receipt; never duplicate it'
    active = read(args.runtime_root/'active-training.json')
    for item in items:
        check_previous_owner(active, item['task'])
        assert not Path(item['plan']['working_directory']).exists()
    physical = check_devices([device for item in items for device in item['devices']])
    args.receipt_dir.mkdir(parents=True)
    for item in items:
        # Recheck ownership and occupancy at this task's actual Popen boundary.
        physical = check_devices(item['devices'])
        receipt = submit_one(args.runtime_root, item, args.receipt_dir, physical)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
