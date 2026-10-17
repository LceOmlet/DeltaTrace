"""Check the installed PyTorch allocator API in a disposable process.

No model, DT, optimizer, GPU tensor or training process is used. Keep old
host-allocated blocks live while allocating registered blocks, then let the
original allocator free both. This checks the exact mixed-config lifecycle
documented by PyTorch 2.8, rather than assuming a vendor fork supports it.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
DEST = AUDIT / 'installed-native-pinned-allocator-behavior-v2.json'


def settings(register):
    return ('expandable_segments:False,pinned_use_cuda_host_register:'
            + str(register) + ',pinned_num_register_threads:1')


def main():
    assert not DEST.exists(), 'Retain the prior behavior result'
    torch.set_num_threads(8)
    record = dict(started=time.time(), pid=os.getpid(), torch_version=torch.__version__,
        torch_git_version=torch.version.git_version, source=__file__,
        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        original_allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        GPU_tensor_allocations=0, model_calls=0, DT_calls=0, optimizer_updates=0,
        training_mutations=0, payload_bytes=16 * 2**20)
    try:
        source = torch.arange(4 * 2**20, dtype=torch.int32)
        torch.cuda.memory._set_allocator_settings(settings(False))
        before = dict(torch.cuda.memory.host_memory_stats())
        tick = time.perf_counter()
        old = torch.empty_like(source, pin_memory=True)
        old.copy_(source)
        record['original_block'] = dict(pinned=old.is_pinned(),
            equal=torch.equal(old, source), seconds=time.perf_counter()-tick)
        torch.cuda.memory._set_allocator_settings(settings(True))
        tick = time.perf_counter()
        new = torch.empty_like(source, pin_memory=True)
        new.copy_(source)
        record['registered_block_with_old_still_live'] = dict(pinned=new.is_pinned(),
            equal=torch.equal(new, source), seconds=time.perf_counter()-tick)
        del old, new
        torch._C._host_emptyCache()
        after = dict(torch.cuda.memory.host_memory_stats())
        counter_names = ('host_alloc_time.total', 'host_free_time.total', 'num_host_alloc', 'num_host_free')
        record['native_counters_delta'] = {key: after[key]-before[key] for key in
            counter_names if key in before and key in after}
        record['native_counters_unavailable'] = [key for key in counter_names
            if key not in before or key not in after]
        record['counter_caveat'] = ('The official wrapper returns an empty mapping before '
            'torch CUDA initialization; unavailable counters are not zero.')
        record['native_counter_time_unit'] = 'microseconds'
        assert record['original_block']['pinned'] and record['original_block']['equal']
        assert record['registered_block_with_old_still_live']['pinned']
        assert record['registered_block_with_old_still_live']['equal']
        record['status'] = 'installed_API_mixed_allocation_and_original_free_passed'
    except BaseException as exc:
        record.update(status='installed_API_behavior_failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(finished=time.time(), torch_CUDA_initialized=torch.cuda.is_initialized())
        DEST.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
