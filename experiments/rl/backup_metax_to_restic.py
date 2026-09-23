"""Back up owner artifacts directly from MetaX to restic and verify off-machine restore.

This orchestrates SSH and restic. It neither creates training checkpoints
nor implements backup storage, compression, encryption or deduplication.
The connector is a local credential provider exposing connect() -> SSHClient.
"""
import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connector', type=Path, required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--destination', default='liangchen@10.70.5.230')
    p.add_argument('--destination-port', default='2501')
    p.add_argument('--jump', default='4090')
    p.add_argument('--backup-root', default='/data/liangchen/deltatrace_rl_metax_backup')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--checkpoint', help='Optional completed checkpoint probe relative to source-root')
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('local_ssh_credentials', args.connector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.connect()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-J', args.jump, '-p', args.destination_port, args.destination]
    b = PurePosixPath(args.backup_root)
    restic = [str(b/'bin/restic'), '-r', str(b/'repository'), '--password-file', str(b/'repository.password'),
              '--cache-dir', str(b/'cache')]
    access = PurePosixPath(args.source_root)/'backup-access'
    remote_restic = [str(access/'restic'), '-r', 'sftp:backup-a6000:'+str(b/'repository'),
                     '--password-file', str(access/'repository.password'),
                     '--cache-dir', str(access/'cache'),
                     '-o', 'sftp.command=ssh -F '+str(access/'ssh_config')+' backup-a6000 -s sftp']
    report = {'source_root': args.source_root, 'destination': args.destination,
              'backup_root': args.backup_root, 'started': time.time(), 'transport': 'MetaX direct SFTP through 4090; no PC data relay', 'snapshots': []}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def destination(cmd):
        return subprocess.check_output(ssh+[shlex.join(cmd)], text=True)

    def record():
        args.receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    try:
        previous = json.loads(destination(restic+['snapshots', '--json']))
        verified_labels = {tag for snap in previous
                           if 'dt-checkpoint-restore-sha256' in snap.get('tags', [])
                           for tag in snap.get('tags', [])}
        with client.open_sftp() as sftp:
            entries = [x for x in ('repo/experiments/rl', 'receipts', 'runs', 'formal-training.json',
                                   'active-training.json', 'active-source.json', 'environment.json')
                       if exists(sftp, str(PurePosixPath(args.source_root)/x))]
            jobs = []
            manifest = str(PurePosixPath(args.source_root)/'formal-training.json')
            if exists(sftp, manifest):
                with sftp.open(manifest) as f:
                    training = json.load(f)
                jobs = training['jobs']
                # Current jobs load an immutable release rather than repo/.
                # Archive that exact source and its environment/import receipt.
                if training.get('dt_root'):
                    release = PurePosixPath(training['dt_root']).relative_to(args.source_root)
                    if str(release) not in entries:
                        entries.append(str(release))
            ray_logs = []
            for job in jobs:
                # These are the exact per-job owner log roots from the launch
                # manifest. Resolve session_latest so restic archives files,
                # rather than only the symlink, without traversing other jobs.
                ray_root = PurePosixPath(job['ray_tmpdir'])
                latest = ray_root/'ray/session_latest/logs'
                if exists(sftp, str(latest)):
                    resolved = PurePosixPath(sftp.normalize(str(latest)))
                    resolved.relative_to(ray_root)
                    ray_logs.append(str(resolved))
            # The official AppWorld client writes API traces/evaluation files
            # alongside its configured port file, outside trainer rollouts.
            for job in jobs:
                port_file = job.get('settings', {}).get('APPWORLD_PORT_FILE')
                if job['task'] == 'AppWorld' and port_file:
                    outputs = PurePosixPath(port_file).parent/'experiments/outputs'
                    if exists(sftp, str(outputs)):
                        entries.append(str(outputs.relative_to(args.source_root)))
            snapshots = [('metadata', entries, False)]
            for job in jobs:
                root = PurePosixPath(job['checkpoint_dir'])
                marker = str(root/'latest_checkpointed_iteration.txt')
                if not exists(sftp, marker):
                    continue
                with sftp.open(marker) as f:
                    latest = int(f.read())
                for name in sorted(sftp.listdir(str(root))):
                    if not name.startswith('global_step_'):
                        continue
                    step = int(name.removeprefix('global_step_'))
                    label = f"{PurePosixPath(job['run_dir']).name}-step-{step}"
                    if step <= latest and label not in verified_labels:
                        rel = str((root/name).relative_to(args.source_root))
                        snapshots.append((label, [rel], True))
            if args.checkpoint:
                label = 'probe-'+PurePosixPath(args.checkpoint).name
                if label not in verified_labels:
                    snapshots.append((label, [args.checkpoint], True))
        for label, paths, immutable in snapshots:
            for path in paths:
                if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                    raise ValueError('Backup sources must remain within the recorded runtime')
            log = args.receipt.with_name(args.receipt.stem+'-'+label+'.log')
            cmd = remote_restic+['backup', '--host', 'deltatrace-metax',
                                 '--tag', 'dt-rl-direct', '--tag', label, '--json']
            if not immutable:
                cmd += ['--exclude=*.pt', '--exclude=__pycache__', '--exclude=*/checkpoints',
                        '--exclude=*/checkpoint-owner-roundtrip*']
            absolute_paths = [str(PurePosixPath(args.source_root)/path) for path in paths]
            if not immutable:
                absolute_paths += ray_logs
                report['metadata_sources'] = absolute_paths
            cmd += ['--', *absolute_paths]
            print('Direct backup', label, flush=True)
            _, output, errors = client.exec_command(shlex.join(cmd))
            with log.open('w', encoding='utf-8') as stream:
                for line in output:
                    stream.write(line)
                    stream.flush()
                error_text = errors.read().decode()
                stream.write(error_text)
            status = output.channel.recv_exit_status()
            if status:
                raise RuntimeError(f'{label}: restic exit {status}: {error_text}')
            summary = next(json.loads(line) for line in reversed(log.read_text().splitlines())
                           if line.startswith('{') and json.loads(line).get('message_type') == 'summary')
            snapshot = summary['snapshot_id']
            row = dict(label=label, snapshot_id=snapshot, restore_verified=False, summary=summary)
            report['snapshots'].append(row)
            record()
            # The destination's original restic restores and verifies every file,
            # without sending a second checkpoint through this PC or to MetaX.
            restore_root = b/'restore-check'/snapshot
            restored = destination(restic+['restore', snapshot, '--target', str(restore_root), '--verify'])
            row['restic_restore'] = restored
            if immutable:
                # Compare full source/restored SHA256, including model, optimizer
                # and extra state, in addition to restic's content verification.
                hash_script = """import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out={}
for name in sys.argv[2:]:
 for f in sorted((root/name).rglob('*')):
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
                destination(restic+['tag', '--add', 'dt-checkpoint-restore-sha256', snapshot])
                tagged = json.loads(destination(restic+['snapshots', '--json']))
                verified = [s for s in tagged if label in s.get('tags', [])
                            and 'dt-checkpoint-restore-sha256' in s.get('tags', [])]
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
