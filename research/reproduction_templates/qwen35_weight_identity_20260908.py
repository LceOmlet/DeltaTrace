"""Stream existing weights once and compare with the frozen official Hub LFS hashes."""
import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
M = Path('${PRIVATE_MOUNT_PATH}')
EXPECTED = {
    'model.safetensors-00001-of-00004.safetensors': 'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e',
    'model.safetensors-00002-of-00004.safetensors': '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330',
    'model.safetensors-00003-of-00004.safetensors': '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec',
    'model.safetensors-00004-of-00004.safetensors': 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6',
}
r = {'status': 'running', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
     'official_api_receipt_sha256': 'ef909fe5b1e01dd51381f3ed429395a0ec6a24822d61fbbeaad0adb0ca4cfa5b',
     'study_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'model_calls': 0, 'GPU_calls': 0, 'shards': {}}


def save():
    tmp = A / 'results.partial'
    tmp.write_text(json.dumps(r, indent=2))
    tmp.replace(A / 'results.json')


save()
try:
    for name, expected in EXPECTED.items():
        path = M / name
        before = path.stat()
        start = time.monotonic()
        h = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                h.update(block)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        r['shards'][name] = {'sha256': h.hexdigest(), 'bytes': after.st_size,
                             'seconds': time.monotonic() - start, 'matches_official': h.hexdigest() == expected}
        save()
        assert h.hexdigest() == expected
    r['status'] = 'all_four_shards_match_official_LFS_sha256'
except Exception:
    r['status'] = 'weight_identity_failed'
    r['error'] = traceback.format_exc()
finally:
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__), 'study.py')
        z.write(A / 'results.json', 'results.json')
    print(json.dumps({'status': r['status'], 'completed_shards': len(r['shards'])}), flush=True)
