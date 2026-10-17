"""Shared host-memory lifecycle boundary for the four native training jobs.

Linux owns cross-process exclusion. PyTorch owns pinned-buffer lifetime and
release. This module neither implements training nor changes a tensor value.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


@contextmanager
def host_memory_lease(phase):
    path = os.environ.get('DT_NATIVE_HOST_MEMORY_LEASE')
    if not path:
        yield
        return
    started = time.monotonic()
    with open(path, 'a') as stream:
        # The driver holds the lock, outside the collective worker RPC. A lock
        # in each rank would deadlock the original FSDP collectives.
        fcntl.flock(stream, fcntl.LOCK_EX)
        print('[native_host_memory_lease] ' + json.dumps(dict(
            phase=phase, event='acquired', pid=os.getpid(), collected=time.time(),
            wait_seconds=time.monotonic()-started)), flush=True)
        try:
            yield
        finally:
            print('[native_host_memory_lease] ' + json.dumps(dict(
                phase=phase, event='released', pid=os.getpid(), collected=time.time())), flush=True)
            fcntl.flock(stream, fcntl.LOCK_UN)


def release_unused_host_cache(phase):
    if os.environ.get('DT_NATIVE_HOST_CACHE_BATCH') != '1':
        return
    import torch
    before = torch.cuda.memory.host_memory_stats()
    started = time.monotonic()
    # This is the same owner API already used at the accepted RPC boundary.
    # It releases inactive cached buffers, retaining all live tensors/events.
    torch._C._host_emptyCache()
    after = torch.cuda.memory.host_memory_stats()
    print('[native_host_cache_batch] ' + json.dumps(dict(
        phase=phase, pid=os.getpid(), collected=time.time(),
        allocated_before=before['allocated_bytes.current'],
        allocated_after=after['allocated_bytes.current'],
        reserved_before=before['reserved_bytes.current'],
        reserved_after=after['reserved_bytes.current'],
        release_seconds=time.monotonic()-started,
        mem_available_KiB=int(next(line.split()[1] for line in
            Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))))), flush=True)
