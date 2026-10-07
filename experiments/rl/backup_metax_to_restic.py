"""Back up owner artifacts directly from MetaX to restic and verify off-machine restore.

This orchestrates SSH and restic. It neither creates training checkpoints
nor implements backup storage, compression, encryption or deduplication.
The connector is a local credential provider exposing connect() -> SSHClient.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


def write_backup_sources(client, access, paths):
    """Pass exact paths through restic's native NUL-separated input interface."""
    payload = b''.join(path.encode('utf-8') + b'\0' for path in paths)
    source_list = str(access/('backup-sources-'+hashlib.sha256(payload).hexdigest()+'.raw'))
    with client.open_sftp() as sftp:
        with sftp.open(source_list, 'wb') as stream:
            stream.write(payload)
    return source_list


def source_ssh_owner():
    """Reuse the existing strict MetaX OpenSSH transport, including keepalives."""
    path = Path(__file__).resolve().parents[2]/'research/temporary/rl_upstream_alignment_20260929/stage_environment_entry.py'
    spec = importlib.util.spec_from_file_location('backup_source_ssh_owner', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return path, owner.SSH


def launch_backup_command(source_ssh, access, command):
    """Use the existing remote owner Popen/log/exit-file lifecycle for one command."""
    script = r'''import json,pathlib,psutil,shlex,subprocess,sys,tempfile
access=pathlib.Path(sys.argv[1]); command=json.loads(sys.argv[2])
directory=pathlib.Path(tempfile.mkdtemp(prefix='restic-command-',dir=access))
run=directory/'run.sh'; log=directory/'output.log'; exit_file=directory/'exit-code'
run.write_text(shlex.join(command)+"\nrc=$?\nprintf '%s\\n' \"$rc\" > "+shlex.quote(str(exit_file))+"\nexit \"$rc\"\n")
with log.open('w') as output:
 proc=subprocess.Popen(['bash',str(run)],stdout=output,stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL,start_new_session=True)
job={'pid':proc.pid,'observed_process_created_unix':psutil.Process(proc.pid).create_time(),
     'directory':str(directory),'log':str(log),'exit_file':str(exit_file),'command':command}
(directory/'job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(job))
'''
    result = subprocess.run(source_ssh+[shlex.join(
        ['/opt/conda/bin/python', '-c', script, str(access), json.dumps(command)])],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def observe_backup_command(source_ssh, job, offset):
    """Read the same detached command; an observation failure never relaunches it."""
    script = r'''import base64,json,pathlib,psutil,sys
job=json.loads(sys.argv[1]); offset=int(sys.argv[2]); log=pathlib.Path(job['log'])
with log.open('rb') as stream:
 stream.seek(offset); data=stream.read(1024*1024); offset=stream.tell()
exit_file=pathlib.Path(job['exit_file']); code=int(exit_file.read_text()) if exit_file.exists() else None
try:
 process=psutil.Process(job['pid'])
 alive=process.create_time()==job['observed_process_created_unix'] and process.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess: alive=False
print(json.dumps({'offset':offset,'data':base64.b64encode(data).decode(),
 'exit_code':code,'alive':alive,'log_drained':offset==log.stat().st_size}))
'''
    result = subprocess.run(source_ssh+[shlex.join(
        ['/opt/conda/bin/python', '-c', script, json.dumps(job), str(offset)])],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def recorded_metadata_sources(client, jobs):
    """Resolve only paths recorded by these jobs and their actual open Ray logs."""
    script = """import json,pathlib,sys,psutil
jobs=json.loads(sys.argv[1]); sources=set()
for job in jobs:
 receipt=job.get('source_receipt')
 if receipt:
  p=pathlib.Path(receipt); source=json.loads(p.read_bytes()); sources.add(str(p.resolve()))
  for name in source.get('source_bindings', {}):
   p=pathlib.Path(name)
   if p.is_file(): sources.add(str(p.resolve()))
  environments={source.get(field, {}).get('DT_ENVIRONMENT_JSON') for field in ('resource_environment','environment')}
  environments.add(source.get('candidate_environment', {}).get('path'))
  for name in environments-{None}:
   p=pathlib.Path(name)
   if p.is_file():
    sources.add(str(p.resolve())); env=json.loads(p.read_bytes())
    library=env.get('qwen35', {}).get('finite_library')
    if library: sources.add(str(pathlib.Path(library).resolve(strict=True)))
 pid=job.get('pid'); birth=job.get('observed_process_created_unix')
 if pid is None or birth is None: continue
 try: process=psutil.Process(pid)
 except psutil.NoSuchProcess: continue
 if process.create_time()!=birth or process.status()==psutil.STATUS_ZOMBIE: continue
 fd=pathlib.Path('/proc')/str(pid)/'fd'
 if fd.is_dir():
  for handle in fd.iterdir():
   try: p=handle.resolve(strict=True)
   except OSError: continue
   for parent in p.parents:
    if parent.name=='logs' and parent.parent.name.startswith('session_'):
     sources.add(str(parent)); break
print(json.dumps(sorted(sources)))
"""
    identities = [{key: job.get(key) for key in
                   ('pid', 'observed_process_created_unix', 'source_receipt')} for job in jobs]
    _, output, errors = client.exec_command(shlex.join(
        ['/opt/conda/bin/python', '-c', script, json.dumps(identities)]))
    raw, error = output.read(), errors.read()
    if output.channel.recv_exit_status():
        raise RuntimeError(error.decode())
    return json.loads(raw)


def select_backup_sources(sftp, source_root, training, verified_labels, checkpoints,
                          profiler_trace, profiler_label):
    """Use either recorded manifest schema, preserving original completion markers."""
    runtime = PurePosixPath(source_root)
    entries = [x for x in ('repo/experiments/rl', 'receipts', 'runs', 'formal-training.json',
                           'active-training.json', 'active-source.json', 'environment.json')
               if exists(sftp, str(runtime/x))]
    jobs = training.get('jobs', [])
    for owner in [training, *jobs]:
        for field in ('dt_root', 'entry', 'verl_root', 'loop_root', 'source_receipt'):
            if owner.get(field):
                relative = str(PurePosixPath(owner[field]).relative_to(runtime))
                if relative not in entries:
                    entries.append(relative)
    ray_logs = []
    for job in jobs:
        if not job.get('ray_tmpdir'):
            continue  # New manifests use actual driver-open logs collected above.
        ray_root = PurePosixPath(job['ray_tmpdir'])
        session = (PurePosixPath(job['ray_session']) if job.get('ray_session')
                   else ray_root/'ray/session_latest')
        latest = session/'logs'
        if exists(sftp, str(latest)):
            resolved = PurePosixPath(sftp.normalize(str(latest)))
            resolved.relative_to(ray_root)
            ray_logs.append(str(resolved))
    for job in jobs:
        port_file = job.get('settings', {}).get('APPWORLD_PORT_FILE')
        if job['task'] == 'AppWorld' and port_file:
            outputs = PurePosixPath(port_file).parent/'experiments/outputs'
            if exists(sftp, str(outputs)):
                entries.append(str(outputs.relative_to(runtime)))
    snapshots = [('metadata', entries, False)]
    for job in jobs:
        root = PurePosixPath(job.get('checkpoint_dir') or job['checkpoints'])
        marker = str(root/'latest_checkpointed_iteration.txt')
        if not exists(sftp, marker):
            continue
        with sftp.open(marker) as f:
            latest = int(f.read())
        for name in sorted(sftp.listdir(str(root))):
            if not name.startswith('global_step_'):
                continue
            step = int(name.removeprefix('global_step_'))
            run = PurePosixPath(job.get('run_dir') or job['output']).relative_to(runtime)
            label = f"{'__'.join(run.parts)}-step-{step}"
            if step <= latest and label not in verified_labels:
                rel = str((root/name).relative_to(runtime))
                snapshots.append((label, [rel], True))
    for checkpoint in checkpoints:
        relative = PurePosixPath(checkpoint)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Checkpoint must remain within the recorded runtime')
        if not relative.name.startswith('global_step_'):
            raise ValueError('Checkpoint must be an original global_step_N directory')
        marker = runtime/relative.parent/'latest_checkpointed_iteration.txt'
        with sftp.open(str(marker)) as source:
            latest = int(source.read())
        if int(relative.name.removeprefix('global_step_')) > latest:
            raise ValueError('Checkpoint is not covered by the original completion marker')
        label = 'checkpoint-'+'__'.join(relative.parts)
        if label not in verified_labels:
            snapshots.append((label, [checkpoint], True))
    # Preserve the legacy artifact, after all currently completed checkpoints.
    if (profiler_label not in verified_labels and exists(sftp, str(runtime/profiler_trace))):
        snapshots.append((profiler_label, [profiler_trace], True))
    return snapshots, ray_logs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connector', type=Path, required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--destination', default='liangchen@10.70.5.230')
    p.add_argument('--destination-port', default='2501')
    p.add_argument('--jump', default='4090')
    p.add_argument('--backup-root', default='/data/liangchen/deltatrace_rl_metax_backup')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--checkpoint', action='append', default=[],
                   help='Additional completed checkpoint relative to source-root; repeat for different runs/tasks')
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('local_ssh_credentials', args.connector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.connect()
    client.get_transport().set_keepalive(15)
    source_owner, source_ssh = source_ssh_owner()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-J', args.jump, '-p', args.destination_port, args.destination]
    b = PurePosixPath(args.backup_root)
    restic = [str(b/'bin/restic'), '-r', str(b/'repository'), '--password-file', str(b/'repository.password'),
              '--cache-dir', str(b/'cache')]
    access = PurePosixPath(args.source_root)/'backup-access'
    remote_restic = [str(access/'restic'), '-r', 'sftp:backup-a6000:'+str(b/'repository'),
                     '--password-file', str(access/'repository.password'),
                     '--cache-dir', str(access/'cache'),
                     '-o', 'sftp.command=ssh -F '+str(access/'ssh_config')+' backup-a6000 -s sftp']
    report = {'status': 'running', 'source_root': args.source_root, 'destination': args.destination,
              'backup_root': args.backup_root, 'started': time.time(), 'transport': 'MetaX direct SFTP through 4090; no PC data relay', 'snapshots': []}
    report['source_ssh_owner'] = {'path': str(source_owner),
                                  'sha256': hashlib.sha256(source_owner.read_bytes()).hexdigest(),
                                  'command': source_ssh}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def destination(cmd):
        return subprocess.check_output(ssh+[shlex.join(cmd)], text=True)

    def record():
        args.receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    record()
    try:
        previous = json.loads(destination(restic+['snapshots', '--json']))
        verified_labels = {tag for snap in previous
                           if {'dt-checkpoint-restore-sha256', 'dt-artifact-restore-sha256'} & set(snap.get('tags', []))
                           for tag in snap.get('tags', [])}
        # This completed native profiler trace is 47GiB. Preserve it in its
        # own verified restic snapshot instead of restoring it every hour as
        # part of mutable logs. Restic owns compression, deduplication and tags.
        profiler_trace = 'receipts/rollout-major-cost/webshop-native-live/generation-206031.json'
        profiler_label = 'native-profiler-webshop-generation-206031'
        with client.open_sftp() as sftp:
            training = {}
            manifest = str(PurePosixPath(args.source_root)/'formal-training.json')
            if exists(sftp, manifest):
                with sftp.open(manifest) as f:
                    training = json.load(f)
            snapshots, ray_logs = select_backup_sources(sftp, args.source_root, training,
                verified_labels, args.checkpoint, profiler_trace, profiler_label)
        # Restic already accepts absolute Ray logs outside source_root. Use the
        # same metadata-only path for recorded resolved imports, including HF.
        covered = [PurePosixPath(args.source_root)/path for path in snapshots[0][1]]
        recorded = recorded_metadata_sources(client, training.get('jobs', []))
        ray_logs = sorted({path for path in ray_logs + recorded
                           if not any(PurePosixPath(path).is_relative_to(parent) for parent in covered)})
        for label, paths, immutable in snapshots:
            for path in paths:
                if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                    raise ValueError('Backup sources must remain within the recorded runtime')
            log = args.receipt.with_name(args.receipt.stem+'-'+label+'.log')
            cmd = remote_restic+['backup', '--host', 'deltatrace-metax',
                                 '--tag', 'dt-rl-direct', '--tag', label, '--json']
            if not immutable:
                cmd += ['--exclude=*.pt', '--exclude=__pycache__', '--exclude=*/checkpoints',
                        '--exclude=*/checkpoint-owner-roundtrip*',
                        '--exclude='+str(PurePosixPath(args.source_root)/profiler_trace)]
            absolute_paths = [str(PurePosixPath(args.source_root)/path) for path in paths]
            if not immutable:
                absolute_paths += ray_logs
                report['metadata_sources'] = absolute_paths
            source_list = write_backup_sources(client, access, absolute_paths)
            cmd += ['--files-from-raw', source_list]
            print('Direct backup', label, flush=True)
            job = launch_backup_command(source_ssh, access, cmd)
            report['active_remote_command'] = dict(label=label, **job)
            record()
            offset = 0
            with log.open('wb') as stream:
                while True:
                    observed = observe_backup_command(source_ssh, job, offset)
                    stream.write(base64.b64decode(observed['data']))
                    stream.flush()
                    offset = observed['offset']
                    if observed['exit_code'] is not None and observed['log_drained']:
                        status = observed['exit_code']
                        report['active_remote_command']['exit_code'] = status
                        record()
                        break
                    if not observed['alive'] and observed['exit_code'] is None:
                        raise RuntimeError(f'{label}: detached command is no longer alive and has no exit file: {job}')
                    time.sleep(5)
            if status:
                raise RuntimeError(f'{label}: detached restic exit {status}; see {log}')
            summary = next(json.loads(line) for line in reversed(log.read_text().splitlines())
                           if line.startswith('{') and json.loads(line).get('message_type') == 'summary')
            snapshot = summary['snapshot_id']
            row = dict(label=label, snapshot_id=snapshot, restore_verified=False, summary=summary)
            report['snapshots'].append(row)
            record()
            # The destination's original restic restores and verifies every file,
            # without sending a second checkpoint through this PC or to MetaX.
            restore_root = b/'restore-check'/snapshot
            free_bytes = int(destination(['python3', '-c',
                'import shutil,sys;print(shutil.disk_usage(sys.argv[1]).free)', str(b)]))
            if free_bytes < summary['total_bytes_processed']:
                raise RuntimeError(f'{label}: restore requires {summary["total_bytes_processed"]} bytes, destination has {free_bytes} free')
            restored = destination(restic+['restore', snapshot, '--target', str(restore_root), '--verify'])
            row['restic_restore'] = restored
            if immutable:
                # Compare full source/restored SHA256, including model, optimizer
                # and extra state, in addition to restic's content verification.
                hash_script = """import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out={}
for name in sys.argv[2:]:
 source=root/name
 for f in [source] if source.is_file() else sorted(source.rglob('*')):
  if f.is_file():
   digest=hashlib.sha256()
   with f.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8*1024*1024), b''): digest.update(chunk)
   out[str(f.relative_to(root))]=digest.hexdigest()
print(json.dumps(out,sort_keys=True))
"""
                remote_python = '/opt/conda/bin/python'
                _, hashes, hash_errors = client.exec_command(shlex.join(
                    [remote_python, '-c', hash_script, args.source_root, *paths]))
                source_hashes = json.loads(hashes.read().decode())
                if hashes.channel.recv_exit_status():
                    raise RuntimeError(hash_errors.read().decode())
                restored_source_root = str(restore_root/args.source_root.lstrip('/'))
                destination_hashes = json.loads(destination(
                    ['python3', '-c', hash_script, restored_source_root, *paths]))
                if not source_hashes or source_hashes != destination_hashes:
                    raise RuntimeError(f'{label}: restored checkpoint files differ from source')
                row['source_restored_sha256'] = source_hashes
            row['restore_verified'] = True
            if immutable:
                # Restic owns the durable marker. Subsequent hourly checks do
                # not transfer/restore an already verified immutable checkpoint.
                verified_tag = ('dt-artifact-restore-sha256' if label == profiler_label
                                else 'dt-checkpoint-restore-sha256')
                destination(restic+['tag', '--add', verified_tag, snapshot])
                tagged = json.loads(destination(restic+['snapshots', '--json']))
                verified = [s for s in tagged if label in s.get('tags', [])
                            and verified_tag in s.get('tags', [])]
                row['original_snapshot_id'] = snapshot
                row['snapshot_id'] = verified[-1]['id']
            record()
            cleanup = """import pathlib,shutil,sys
base=pathlib.Path(sys.argv[1]).resolve(); target=pathlib.Path(sys.argv[2]).resolve()
assert target.parent == base and target.name.isalnum()
shutil.rmtree(target)
"""
            destination(['python3', '-c', cleanup, str(b/'restore-check'), str(restore_root)])
            print('Verified off-machine restore', label, snapshot, flush=True)
        report['repository_check'] = destination(restic+['check'])
        report.update(status='passed', finished=time.time())
        record()
    except Exception as exc:
        report.update(status='failed', error=str(exc), finished=time.time())
        record()
        raise
    finally:
        client.close()


def exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    main()
