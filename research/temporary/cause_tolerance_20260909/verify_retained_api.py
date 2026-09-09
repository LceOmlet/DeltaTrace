"""Verify the actual published Qwen3 retention factory, without rerunning scores."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    folder = HERE / 'qwen3_retained_api'
    r = json.loads((folder / 'results.json').read_bytes())
    assert r['status'] == 'complete' and len(r['cases']) == 2 and len(r['calls']) == 12
    assert r['script_sha256'] == sha((HERE / 'benchmark_qwen3_retained_api.py').read_bytes())
    assert r['vectors_sha256'] == sha((folder / 'vectors.npz').read_bytes())
    assert all(r[k] == 0 for k in ('FT_calls', 'generation_calls', 'metric_calls'))
    manifest = ROOT / 'deltatrace/accelerated/retained_sources.json'
    receipt = r['candidate_sources']
    assert receipt['manifest_sha256'] == sha(manifest.read_bytes())
    assert receipt['base_deferred'] == r['baseline_sources']
    for mapping in (receipt['files'], r['baseline_sources']['files']):
        for n, digest in mapping.items():
            assert sha((ROOT / n).read_bytes()) == digest, n
    for name in ('qwen3_retained_replay.py', 'qwen3_retained_pair.py'):
        assert (HERE / name).read_bytes() == (ROOT / 'deltatrace/accelerated/qwen3' / name).read_bytes()
    z = np.load(folder / 'vectors.npz', allow_pickle=False)
    assert len(z.files) == 12
    for case in r['cases']:
        key = f"{case['dataset']}_{case['index']}"
        names = [f'{p}/{m}/{key}' for p in ('warm', 'r0', 'r1') for m in ('baseline', 'candidate')]
        assert all(np.array_equal(z[names[0]], z[n]) and np.isfinite(z[n]).all() for n in names)
    for c in r['calls']:
        assert c['status'] == 'returned'
        v = c['validation']
        assert v['all_passed'] and v['predicates'] == 828 and v['statistics'] == 36
        assert c['native_layer_replay_calls'] == c['public_FA_calls'] == 36
        if not c['name'].startswith('warm/'):
            assert c['compiler_before'] == c['compiler_after']
    summary = {'status': 'verified', 'result_sha256': sha((folder / 'results.json').read_bytes()),
               'vectors_sha256': r['vectors_sha256'], 'actual_published_factory_DT_calls': 12,
               'all_complete_vectors_equal': True, 'published_sources_byte_identical_to_measured': True,
               'scope': 'NI0/MH0 integration check of retained_qwen3 factory; unchanged clean/deferred baseline, official model and FA; no metric or generation calls.'}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
