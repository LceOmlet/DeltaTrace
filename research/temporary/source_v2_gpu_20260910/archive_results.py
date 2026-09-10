"""Store or restore lossless result artifacts with deterministic compression."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a new directory; never overwrite an existing result'
    args.output.mkdir(parents=True)
    if args.restore:
        manifest = json.loads((args.source / 'manifest.json').read_bytes())
        for row in manifest['files']:
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            compressed = (args.source / (row['path'] + '.gz')).read_bytes()
            assert digest(compressed) == row['compressed_sha256']
            data = gzip.decompress(compressed)
            assert digest(data) == row['sha256'] and len(data) == row['bytes']
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        print(json.dumps({'status': 'restored_and_verified', 'files': len(manifest['files'])}))
        return
    state = json.loads((args.source / 'suite_status.json').read_bytes())
    tasks = [r['dataset'] for r in state['completed_tasks']]
    assert tasks, 'Archive only complete task shards'
    paths = [args.source / name for name in ('suite_status.json', 'plan_identity.json')]
    paths += [p for task in tasks for p in sorted((args.source / task).rglob('*'))
              if p.is_file() and p.suffix in ('.json', '.npz')]
    rows = []
    for path in paths:
        data = path.read_bytes()
        compressed = gzip.compress(data, compresslevel=9, mtime=0)
        relative = path.relative_to(args.source).as_posix()
        destination = args.output / (relative + '.gz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(compressed)
        rows.append({'path': relative, 'bytes': len(data), 'sha256': digest(data),
                     'compressed_bytes': len(compressed), 'compressed_sha256': digest(compressed)})
    manifest = {'format': 'gzip_original_bytes_v1', 'suite_status': state['status'],
                'tasks': tasks, 'files': rows}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'archived', 'files': len(rows),
                      'compressed_bytes': sum(r['compressed_bytes'] for r in rows)}))


if __name__ == '__main__':
    main()
