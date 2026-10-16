"""Preserve an inactive diagnostic artifact while moving its storage out of RAM."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'
source = Path('/dev/shm/verl-cache/c8323a76830bb307ce86c03970c3e0e5/numerical-warm-state-original-VERL')
destination = Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/appworld-20261017/inactive-diagnostic-checkpoint-20261017-v1/numerical-warm-state-original-VERL')
receipt = AUDIT / 'inactive-diagnostic-checkpoint-storage-relocation.json'
assert source.parent.resolve() == Path('/dev/shm/verl-cache/c8323a76830bb307ce86c03970c3e0e5')
assert destination.is_relative_to(Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/appworld-20261017'))
assert source.is_dir() and not source.is_symlink() and not destination.exists()
holders = []
for process in psutil.process_iter():
    try:
        handles = [f.path for f in process.open_files() if str(source) in f.path]
        maps = Path(f'/proc/{process.pid}/maps').read_text()
        if handles or str(source) in maps:
            holders.append(dict(pid=process.pid, birth=process.create_time(), handles=handles))
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, PermissionError):
        pass
assert not holders, holders
files = sorted(p for p in source.rglob('*') if p.is_file())
total_bytes = sum(p.stat().st_size for p in files)
assert total_bytes == 97086114342 and len(files) == 11
destination.parent.mkdir(parents=True, exist_ok=True)
assert psutil.disk_usage(destination.parent).free > total_bytes
record = dict(status='hashing_inactive_diagnostic_checkpoint_before_storage_move',
              started=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
              source=str(source), destination=str(destination), files=len(files), bytes=total_bytes,
              MemAvailable_before_GiB=psutil.virtual_memory().available/2**30,
              active_file_or_mapping_holders=holders, original_method_or_training_changes=0,
              checkpoint_create_calls=0, checkpoint_restore_calls=0)


def write():
    receipt.write_text(json.dumps(record, indent=2) + '\n')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


write()
hashes = {str(p.relative_to(source)):digest(p) for p in files}
record.update(status='moving_preserved_diagnostic_checkpoint_with_GNU_mv', SHA256=hashes)
write()
# GNU mv owns the cross-filesystem copy and removal of its old RAM storage.
# This artifact is not a formal training checkpoint or a compiler cache.
subprocess.run(['mv', '--', str(source), str(destination)], check=True)
source.symlink_to(destination, target_is_directory=True)
record['status'] = 'verifying_preserved_checkpoint_bytes_at_same_logical_path'
write()
actual = {str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
assert actual == hashes
assert source.resolve() == destination.resolve()
record.update(status='passed_preserved_diagnostic_checkpoint_storage_relocation',
              finished=time.time(), SHA256_verified=True, original_logical_path_preserved=True,
              MemAvailable_after_GiB=psutil.virtual_memory().available/2**30,
              diagnostic_artifact_deleted=False, compiler_cache_changes=0)
write()
print(json.dumps(record))
