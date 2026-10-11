"""Include CPU-readable MetaX registered host mappings in the procfs reader.

The original incident's actual GPU weight mappings are ---p and remain excluded.
Readability does not identify an allocation as a particular FSDP tensor.
"""
from pathlib import Path


def readable_regions(pid, minimum_bytes):
    regions = []
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        name = fields[5] if len(fields) == 6 else ''
        begin, end = (int(s, 16) for s in fields[0].split('-'))
        private_CPU = fields[1] == 'rw-p' and name in ('', '[heap]')
        registered_host = fields[1] == 'rw-s' and name.startswith('/dev/dri/renderD')
        if end - begin >= minimum_bytes and (private_CPU or registered_host):
            regions.append(dict(begin=begin, end=end, map_line=line))
    return regions
