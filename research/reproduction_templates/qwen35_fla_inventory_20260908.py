"""Read existing Qwen3.5 weights and backend packages without executing model code."""
import hashlib
import importlib.metadata
import json
import struct
import sys
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = Path('${PRIVATE_MOUNT_PATH}')
MODEL = ROOT / 'models/Qwen3.5-9B'
sha = lambda b: hashlib.sha256(b).hexdigest()
r = {'status': 'running', 'model_calls': 0, 'GPU_calls': 0, 'installed_packages_changed': False,
     'study_sha256': sha(Path(__file__).read_bytes()), 'checkpoint': str(MODEL),
     'files': {}, 'shards': {}, 'wheels': {}, 'runtime_packages': {}}

for name in ['config.json', 'model.safetensors.index.json', 'tokenizer_config.json',
             'tokenizer.json', 'chat_template.jinja']:
    data = (MODEL / name).read_bytes()
    r['files'][name] = {'sha256': sha(data), 'bytes': len(data)}
config = json.loads((MODEL / 'config.json').read_text())
index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
r['config'] = config
found = {}
for name in sorted(set(index['weight_map'].values())):
    assert Path(name).name == name
    path = MODEL / name
    with path.open('rb') as f:
        header_length = struct.unpack('<Q', f.read(8))[0]
        assert 0 < header_length < 16 * 1024 * 1024
        header = f.read(header_length)
    tensors = json.loads(header)
    payload_bytes = path.stat().st_size - 8 - header_length
    intervals = []
    for key, info in tensors.items():
        if key == '__metadata__':
            continue
        assert key not in found and index['weight_map'][key] == name
        begin, end = info['data_offsets']
        assert 0 <= begin <= end <= payload_bytes
        intervals.append((begin, end))
        found[key] = {'shape': info['shape'], 'dtype': info['dtype'], 'shard': name}
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == payload_bytes
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    r['shards'][name] = {'bytes': path.stat().st_size, 'header_sha256': sha(header),
                         'header_bytes': header_length, 'tensors': len(intervals),
                         'payload_sha256': None, 'full_weight_identity_verified': False}
assert set(found) == set(index['weight_map'])
r['tensor_metadata'] = found
r['index_metadata'] = index.get('metadata')

selected = [ROOT / 'wheels/tf513_full/transformers-5.13.0-py3-none-any.whl']
selected += sorted((ROOT / 'wheels/veomni').glob('fla_core-*.whl'))
selected += sorted((ROOT / 'wheels/veomni').glob('flash_linear_attention-*.whl'))
for path in selected:
    data = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        metadata_name = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
        metadata = z.read(metadata_name).decode()
        r['wheels'][path.name] = {'sha256': sha(data), 'bytes': len(data),
            'requirements': [s for s in metadata.splitlines() if s.startswith(('Requires-Dist:', 'Requires-Python:', 'Version:', 'Name:'))]}
        if path.name.startswith('transformers-'):
            for n in ['transformers/models/qwen3_5/modeling_qwen3_5.py',
                      'transformers/models/qwen3_5/configuration_qwen3_5.py']:
                out = A / Path(n).name
                out.write_bytes(z.read(n))
                r['wheels'][path.name].setdefault('source_sha256', {})[n] = sha(out.read_bytes())
for name in ['torch', 'triton', 'transformers', 'flash-attn', 'flash-linear-attention', 'fla-core', 'causal-conv1d']:
    try:
        r['runtime_packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        r['runtime_packages'][name] = None
r['python'] = sys.version
r['status'] = 'header_index_and_package_inventory_complete_not_runtime_validated'
(A / 'results.json').write_text(json.dumps(r, indent=2))
with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for file in [Path(__file__), A / 'results.json', A / 'modeling_qwen3_5.py', A / 'configuration_qwen3_5.py']:
        z.write(file, file.name)
print(json.dumps({'status': r['status'], 'shards': len(r['shards']), 'tensors': len(found),
                  'runtime_packages': r['runtime_packages'], 'wheels': r['wheels']}))
