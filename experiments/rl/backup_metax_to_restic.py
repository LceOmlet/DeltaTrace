"""Stream owner artifacts to an off-machine restic repository and verify restore.

This orchestrates SSH, tar and restic. It neither creates training checkpoints
nor implements backup storage, compression, encryption or deduplication.
The connector is a local credential provider exposing connect() -> SSHClient.
"""
import argparse
import hashlib
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
    report = {'source_root': args.source_root, 'destination': args.destination,
              'backup_root': args.backup_root, 'started': time.time(), 'snapshots': []}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def destination(cmd):
        return subprocess.check_output(ssh+[shlex.join(cmd)], text=True)

    def record():
        args.receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    try:
        previous = json.loads(destination(restic+['snapshots', '--json']))
        tags = {tag for snap in previous for tag in snap.get('tags', [])}
        with client.open_sftp() as sftp:
            entries = [x for x in ('repo/experiments/rl', 'receipts', 'runs', 'formal-training.json',
                                   'active-training.json', 'active-source.json', 'environment.json')
                       if exists(sftp, str(PurePosixPath(args.source_root)/x))]
            jobs = []
            manifest = str(PurePosixPath(args.source_root)/'formal-training.json')
            if exists(sftp, manifest):
                with sftp.open(manifest) as f:
                    jobs = json.load(f)['jobs']
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
                    label = f"{job['task']}-step-{step}"
                    if step <= latest and label not in tags:
                        rel = str((root/name).relative_to(args.source_root))
                        snapshots.append((label, [rel], True))
            if args.checkpoint:
                snapshots.append(('checkpoint-roundtrip-probe', [args.checkpoint], True))
        for label, paths, immutable in snapshots:
            if immutable and label in tags:
                continue
            for path in paths:
                if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                    raise ValueError('Backup sources must remain within the recorded runtime')
            tar = ['tar', '-C', args.source_root]
            if not immutable:
                tar += ['--exclude=*.pt', '--exclude=__pycache__', '--exclude=*/checkpoints',
                        '--exclude=*/checkpoint-owner-roundtrip', '--exclude=*/checkpoint-owner-roundtrip-v2',
                        '--exclude=*/checkpoint-owner-roundtrip-v3']
            tar += ['-cf', '-', '--', *paths]
            _, stream, errors = client.exec_command(shlex.join(tar))
            filename = label+'.tar'
            log = args.receipt.with_name(args.receipt.stem+'-'+label+'.log')
            cmd = restic+['backup', '--stdin', '--stdin-filename', filename, '--host', 'deltatrace-metax',
                          '--tag', 'dt-rl', '--tag', label, '--json']
            digest = hashlib.sha256()
            size = 0
            print('Backing up', label, flush=True)
            with log.open('wb') as output:
                proc = subprocess.Popen(ssh+[shlex.join(cmd)], stdin=subprocess.PIPE, stdout=output, stderr=output)
                try:
                    while chunk := stream.read(4*1024*1024):
                        digest.update(chunk)
                        size += len(chunk)
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                    status = proc.wait()
                finally:
                    if proc.poll() is None:
                        proc.terminate()
            source_status = stream.channel.recv_exit_status()
            source_stderr = errors.read().decode()
            if status or source_status:
                raise RuntimeError(f'{label}: restic={status}, source tar={source_status}: {source_stderr}')
            summary = next(json.loads(line) for line in reversed(log.read_text().splitlines())
                           if line.startswith('{') and json.loads(line).get('message_type') == 'summary')
            snapshot = summary['snapshot_id']
            # Verify every archived byte by reading it back through restic on
            # the other host. No second copy of a model lands on this PC.
            verify_cmd = shlex.join(restic+['dump', snapshot, '/'+filename])+' | sha256sum'
            restored_hash = subprocess.check_output(ssh+['bash -o pipefail -c '+shlex.quote(verify_cmd)], text=True).split()[0]
            if restored_hash != digest.hexdigest():
                raise RuntimeError(f'{label}: restored archive checksum differs')
            report['snapshots'].append(dict(label=label, snapshot_id=snapshot, archive_bytes=size,
                archive_sha256=restored_hash, restore_verified=True, summary=summary))
            record()
            print('Verified restore', label, snapshot, size, flush=True)
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
