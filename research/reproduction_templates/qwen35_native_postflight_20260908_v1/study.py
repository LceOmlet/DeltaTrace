"""Read-only source and compiler-resource audit after the bounded native run."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = Path('${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v3')
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (PARENT / 'results.json').read_bytes()
assert sha(raw) == 'f745bdc61ddd9fb305c487512737028d256b8c3630ac2a86d409072944d6e3ae'
d = json.loads(raw); p = d['protocol']
assert d['status'] == 'failed' and d['root_forwards_completed'] == 3
site = Path(p['isolated_site']); count = 0; changed = []; h = hashlib.sha256()
for path, digest in p['wheels'].items():
    raw = Path(path).read_bytes(); assert sha(raw) == digest
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if name.endswith('.py') and name.startswith(('fla/', 'transformers/')):
                actual, original = (site / name).read_bytes(), z.read(name)
                if actual != original:
                    expected = {'fla/utils.py': p['fla_mapped_utils_sha256'],
                                'fla/ops/common/chunk_delta_h.py': p['fla_scheduled_chunk_sha256']}
                    assert name in expected and sha(actual) == expected[name]
                    changed.append(name)
                h.update(name.encode() + bytes.fromhex(sha(actual))); count += 1
assert count == 2853 and len(changed) == 2
assert h.hexdigest() == d['sources_during']['source_tree_sha256']
checkpoint = Path(p['checkpoint'])
stats = {x.name: {'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns} for x in checkpoint.glob('*.safetensors')}
assert stats == d['checkpoint_stats_before']
resources = []
for file in Path(p['compiler_cache']).rglob('chunk_gated_delta_rule_fwd_kernel_h_blockdim64.json'):
    raw = file.read_bytes(); meta = json.loads(raw)
    resources.append({'file': str(file), 'sha256': sha(raw), 'metadata': meta})
r = {'status': 'postflight_sources_and_weight_stats_unchanged', 'study_sha256': sha(Path(__file__).read_bytes()),
     'parent_raw_sha256': sha((PARENT / 'results.json').read_bytes()), 'files_compared': count,
     'explicit_changes': changed, 'source_tree_sha256': h.hexdigest(), 'checkpoint_stats': stats,
     'compiled_state_kernel_variants': resources,
     'resource_scope': 'Compiler cache includes pre-failure and scheduled variants. Metadata alone does not identify the actual chosen configuration for each profiled launch.',
     'model_calls': 0, 'GPU_calls': 0, 'torch_imported': 'torch' in sys.modules}
assert not r['torch_imported']
(HERE / 'results.json').write_text(json.dumps(r, indent=2))
with zipfile.ZipFile(HERE / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ['study.py', 'results.json']: z.write(HERE / name, name)
print(json.dumps({'status': r['status'], 'compiled_variants': len(resources)}))
