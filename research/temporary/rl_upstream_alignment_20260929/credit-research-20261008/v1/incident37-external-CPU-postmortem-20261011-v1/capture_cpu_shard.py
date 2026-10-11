"""Read stopped incident workers through Linux procfs; never import in workers.

This is a diagnostic, not a training implementation. Matching CPU allocations
are candidates, not proof of the particular FSDP tensor identity. No CUDA APIs,
model calls, or numerical acceptance thresholds are introduced.
"""
import hashlib
import json
import os
from pathlib import Path
import time


SHARD_BYTES = 124160 * 4096 * 2
ROW_BYTES = 4096 * 2
LOCAL_ROW = 159048 - 124160
ANCHOR_ROWS = (0, 1, 65536, 124159)
CHUNK = 16 << 20


def anonymous_regions(pid):
    regions = []
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        name = fields[5] if len(fields) == 6 else ''
        begin, end = (int(s, 16) for s in fields[0].split('-'))
        if fields[1] != 'rw-p' or name not in ('', '[heap]'):
            continue
        if end - begin >= SHARD_BYTES:
            regions.append(dict(begin=begin, end=end, map_line=line))
    return regions


def capture(pid, anchors, destination):
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    maps = Path(f'/proc/{pid}/maps').read_text()
    (destination / 'maps.txt').write_text(maps)
    (destination / 'status.txt').write_text(Path(f'/proc/{pid}/status').read_text())
    result = dict(pid=pid, unix=time.time(), candidates=[], errors=[],
                  bytes_read=0, production_changes=0, GPU_calls=0,
                  identity_scope='Multi-anchor matching CPU allocation; not a Python/FSDP object identity')
    regions = anonymous_regions(pid)
    result['eligible_regions'] = regions
    observed = set()
    with open(f'/proc/{pid}/mem', 'rb', buffering=0) as memory:
        def read(address, size):
            data = os.pread(memory.fileno(), size, address)
            result['bytes_read'] += len(data)
            return data

        def inspect_chunk(region, address, size):
            try:
                data = read(address, size)
            except OSError as error:
                result['errors'].append(dict(address=address, size=size, error=repr(error)))
                return
            offset = data.find(anchors[0])
            while offset >= 0:
                base = address + offset
                if base not in observed and base + SHARD_BYTES <= region['end']:
                    observed.add(base)
                    checks = []
                    for row in ANCHOR_ROWS:
                        actual = read(base + row * ROW_BYTES, len(anchors[row]))
                        checks.append(dict(local_row=row, exact=actual == anchors[row]))
                    if all(check['exact'] for check in checks):
                        row = read(base + LOCAL_ROW * ROW_BYTES, ROW_BYTES)
                        path = destination / f'candidate-{len(result["candidates"])}-row159048.bin'
                        path.write_bytes(row)
                        result['candidates'].append(dict(base=base, row_pointer=base + LOCAL_ROW * ROW_BYTES,
                            anchor_checks=checks, row_path=str(path), row_sha256=hashlib.sha256(row).hexdigest(),
                            row_bytes=len(row), map_line=region['map_line']))
                offset = data.find(anchors[0], offset + 1)

        # Big CPU allocations usually begin near a VMA boundary. Examine this
        # cheap part first, but do not treat no match here as an exclusion.
        for region in regions:
            inspect_chunk(region, region['begin'], min(4 << 20, region['end'] - region['begin']))
        if not result['candidates']:
            for region in regions:
                for address in range(region['begin'], region['end'], CHUNK):
                    inspect_chunk(region, address, min(CHUNK + len(anchors[0]) - 1, region['end'] - address))
    result['seconds'] = time.perf_counter() - started
    (destination / 'CPU-shard-candidates.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
