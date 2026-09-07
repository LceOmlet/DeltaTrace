"""Read pinned compiled ELF metadata; no library loading or GPU execution."""
import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import msgpack

LIBRARIES = {
    'shared_mean_reuse': (
        '${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1/libdeltatrace_fa_finite_shared_mean_reuse.so',
        'd5e0cac33f206fcbd0a3c31364fea06cc37b39165dea7c571dfd40519119665b'),
    'two_sweeps': (
        '${ARTIFACT_ROOT}/codex_fa_two_sweeps_build_20260907_v4/libdeltatrace_fa_finite_two_sweeps.so',
        'bcd9d5103df38957fa6112e6fe626b81e9b045a54c296509300c9b24108b3f03'),
}
sha = lambda b: hashlib.sha256(b).hexdigest()


def sections(data):
    assert data[:6] == b'\x7fELF\x02\x01', 'Requires ELF64 little endian'
    start = struct.unpack_from('<Q', data, 40)[0]
    size, count, names_index = struct.unpack_from('<HHH', data, 58)
    assert size == 64 and names_index < count and start + size * count <= len(data)
    rows = [struct.unpack_from('<IIQQQQIIQQ', data, start + i * size) for i in range(count)]
    name_row = rows[names_index]
    names = data[name_row[4]:name_row[4] + name_row[5]]
    for row in rows:
        name = names[row[0]:].split(b'\0', 1)[0].decode()
        if row[1] != 8:  # SHT_NOBITS has no file payload.
            assert row[4] + row[5] <= len(data)
            yield name, row[1], data[row[4]:row[4] + row[5]]


def bundles(data):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    assert data.startswith(magic)
    pos = len(magic)
    count = struct.unpack_from('<Q', data, pos)[0]
    pos += 8
    for _ in range(count):
        offset, size, name_size = struct.unpack_from('<QQQ', data, pos)
        pos += 24
        name = data[pos:pos + name_size].decode()
        pos += name_size
        assert offset + size <= len(data)
        yield name, data[offset:offset + size]


def notes(data):
    pos = 0
    while pos < len(data):
        name_size, desc_size, kind = struct.unpack_from('<III', data, pos)
        pos += 12
        name = data[pos:pos + name_size].rstrip(b'\0').decode()
        pos += (name_size + 3) & ~3
        desc = data[pos:pos + desc_size]
        assert len(desc) == desc_size
        pos += (desc_size + 3) & ~3
        yield name, kind, desc
    assert pos == len(data)


def inspect(path, expected):
    data = path.read_bytes()
    assert sha(data) == expected
    fatbins = [payload for name, _, payload in sections(data) if name == '.mc_fatbin']
    assert len(fatbins) == 1
    images = []
    for target, payload in bundles(fatbins[0]):
        if not payload.startswith(b'\x7fELF'):
            continue
        metadata = []
        for _, kind, section in sections(payload):
            if kind != 7:
                continue
            for owner, note_type, desc in notes(section):
                if owner == 'MetaX':
                    decoded = msgpack.unpackb(desc, raw=False, strict_map_key=True)
                    assert isinstance(decoded, dict) and 'macahca.kernels' in decoded
                    metadata.append({'owner': owner, 'note_type': note_type,
                                     'descriptor_sha256': sha(desc), 'decoded': decoded})
        assert metadata
        images.append({'target': target, 'sha256': sha(payload), 'metadata': metadata})
    assert len(images) == 1
    return {'library_sha256': expected, 'bytes': len(data), 'images': images}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'compiled_metadata_read_only',
              'study_sha256': sha(Path(__file__).read_bytes()),
              'msgpack_version': msgpack.__version__,
              'GPU_calls': 0, 'model_calls': 0, 'compiles': 0,
              'libraries': {name: inspect(Path(path), expected)
                            for name, (path, expected) in LIBRARIES.items()},
              'limits': 'Static compiler metadata; not measured occupancy, memory traffic or latency attribution.'}
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    with zipfile.ZipFile(output / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(output / 'results.json', 'results.json')
        z.write(Path(__file__), 'study.py')
    for name, row in result['libraries'].items():
        for meta in row['images'][0]['metadata']:
            for kernel in meta['decoded']['macahca.kernels']:
                print(name, json.dumps({k: v for k, v in kernel.items() if k != '.args'}))


if __name__ == '__main__':
    main()
