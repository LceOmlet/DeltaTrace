"""Backport vLLM's merged fresh-page initialization to the MetaX owner boundary.

Upstream: vllm-project/vllm#44972, merge
0033211c0bcf3d42d3b1f27059922c30e664b8e7, csrc/cumem_allocator.cpp.
That implementation is ROCm-only. MetaX's installed binary is not rebuilt:
its existing Python create_and_map boundary calls the existing checked
CudaRTLibrary.cudaMemset (mapped by the plugin to mcMemset), on the handle's
device, after the native mapping. Sleep/wake, tags, weight backup and restoration
are unchanged. This is a platform adaptation, not an unchanged upstream binary.
Apply explicitly only after reproducing stale-state generation on this runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path

OLD = '''def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)
'''
NEW = OLD + '''    # vLLM #44972: newly mapped hybrid-state pages must not contain stale NaNs.
    # MetaX's checked owner wrapper maps cudaMemset to mcMemset.
    with torch.cuda.device(allocation_handle[0]):
        libcudart.cudaMemset(allocation_handle[2], 0, allocation_handle[1])
'''


def patch_source(source):
    if NEW in source:
        return source
    if source.count(OLD) != 1:
        raise RuntimeError('Installed MetaX create_and_map differs from the recorded owner')
    return source.replace(OLD, NEW, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    before = args.source.read_text()
    after = patch_source(before)
    backup = args.receipt.with_suffix('.before.py')
    if after != before:
        if backup.exists():
            raise RuntimeError(f'Preserve existing receipt before applying another revision: {backup}')
        backup.write_text(before)
        args.source.write_text(after)
    args.receipt.write_text(json.dumps(dict(source=str(args.source), backup=str(backup),
        upstream='https://github.com/vllm-project/vllm/commit/0033211c0bcf3d42d3b1f27059922c30e664b8e7',
        adaptation=__doc__, before_sha256=hashlib.sha256(before.encode()).hexdigest(),
        after_sha256=hashlib.sha256(after.encode()).hexdigest(), changed=after != before,
        validation='pending real generation replay'), indent=2)+'\n')
    print(f'Owner mapping initialization: {args.source}; changed={after != before}')
