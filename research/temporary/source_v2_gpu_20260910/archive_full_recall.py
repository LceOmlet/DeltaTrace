"""Archive exactly the completed full-run artifacts, retaining original bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--zip', type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'Preserve existing archives'
    if a.zip:
        assert not a.zip.exists() and not a.zip.with_suffix('.receipt.json').exists()
    plan = json.loads(a.plan.read_bytes())
    state = json.loads((a.source / 'progress.json').read_bytes())
    identity = json.loads((a.source / 'execution_identity.json').read_bytes())
    assert state['status'] == 'complete'
    assert state['new_completed'] == 368 and state['control_completed'] == 5
    assert state['identity'] == identity and identity['plan_sha256'] == sha(a.plan.read_bytes())
    receipts = {x['chunk']: x for x in state['completed_chunks']}
    assert len(receipts) == len(state['completed_chunks']) == len(plan['chunks']) == 26
    assert set(receipts) == set(plan['chunks'])
    paths = ['progress.json', 'execution_identity.json']
    for name, chunk in plan['chunks'].items():
        assert Path(name).name == name and name not in ('.', '..')
        folder = a.source / name
        raw = (folder / 'results.json').read_bytes()
        result = json.loads(raw)
        assert result['status'] == 'complete' and result['chunk_spec'] == chunk
        assert sha(raw) == receipts[name]['results_sha256']
        assert sha((folder / 'vectors.npz').read_bytes()) == receipts[name]['vectors_sha256'] == result['vectors_sha256']
        paths += [name + '/results.json', name + '/vectors.npz']
    a.output.mkdir(parents=True)
    rows = []
    for relative in paths:
        data = (a.source / relative).read_bytes()
        compressed = gzip.compress(data, compresslevel=9, mtime=0)
        assert gzip.decompress(compressed) == data
        destination = a.output / (relative + '.gz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(compressed)
        rows.append(dict(path=relative, bytes=len(data), sha256=sha(data),
                         compressed_bytes=len(compressed), compressed_sha256=sha(compressed)))
    manifest = dict(format='gzip_original_bytes_v1', suite_status='complete',
                    full_run=True, unique_cases=448, new_cases=368, reused_cases=80,
                    control_cases=5, tasks=list(plan['tasks']),
                    plan_sha256=identity['plan_sha256'],
                    parent_results_sha256=plan['parent_results_sha256'],
                    parent_vectors_sha256=plan['parent_vectors_sha256'], files=rows)
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    receipt = dict(status='archived_and_roundtrip_verified', files=len(rows),
                   original_bytes=sum(x['bytes'] for x in rows),
                   compressed_bytes=sum(x['compressed_bytes'] for x in rows),
                   manifest_sha256=sha((a.output / 'manifest.json').read_bytes()))
    if a.zip:
        with zipfile.ZipFile(a.zip, 'x', compression=zipfile.ZIP_STORED) as z:
            for relative in ['manifest.json'] + [x['path'] + '.gz' for x in rows]:
                z.write(a.output / relative, relative)
        receipt.update(zip_sha256=sha(a.zip.read_bytes()), zip_bytes=a.zip.stat().st_size)
        a.zip.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
