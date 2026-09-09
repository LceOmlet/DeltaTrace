"""Audit saved native replay-retention experiments without model or score calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    args = p.parse_args()
    ref = args.snapshots / 'tmp/codex_clean_development16_20260909_v1/qwen3/results.json'
    assert sha(ref.read_bytes()) == 'dccfbf8d2f2ba33b0ff68c686031fb86b2ac76501164f63d91228be545984de6'
    extra = HERE / 'qwen3_batch_extra_inputs.json'
    assert sha(extra.read_bytes()) == '28c513b607d30d7275063649cc08699971cc3cece76002dad5e2caf7ba0f547e'
    reference = {f"{c['dataset']}_{c['index']}": c for path in (ref, extra)
                 for c in json.loads(path.read_bytes())['cases']}
    derivation = json.loads((HERE / 'qwen3_retained_derivation.json').read_bytes())
    for r in derivation['files']:
        assert sha((ROOT / r['source']).read_bytes()) == r['source_sha256']
        raw = (ROOT / r['source']).read_bytes()
        if r['output'].endswith('replay.py'):
            assert raw.count(b'.detach().clone(memory_format=torch.preserve_format)') == 4
            raw = raw.replace(b'.detach().clone(memory_format=torch.preserve_format)', b'.detach()')
        else:
            raw = raw.replace(b'from qwen3_deferred_replay import', b'from qwen3_retained_replay import')
        assert raw == (HERE / r['output']).read_bytes()
        assert sha(raw) == r['output_sha256']
    summaries = {}
    for job, phase, count in [('qwen3_retained_pilot', 'pilot', 4),
                              ('qwen3_retained16', 'development16', 16),
                              ('qwen3_retained_extra', 'extra6', 6)]:
        path = HERE / job
        r = json.loads((path / 'results.json').read_bytes())
        assert r['status'] == 'complete' and r['phase'] == phase
        assert r['script_sha256'] == sha((HERE / 'benchmark_qwen3_retained.py').read_bytes())
        assert all(r[k] == 0 for k in ('FT_calls', 'generation_calls', 'metric_calls'))
        for f, digest in r['baseline_sources']['files'].items():
            assert sha((ROOT / f).read_bytes()) == digest, f
        for f, digest in r['candidate_sources'].items():
            assert sha((HERE / f).read_bytes()) == digest, f
        assert r['vectors_sha256'] == sha((path / 'vectors.npz').read_bytes())
        z = np.load(path / 'vectors.npz', allow_pickle=False)
        assert len(r['cases']) == count and len(r['calls']) == len(z.files) == 6 * count
        calls = {c['name']: c for c in r['calls']}
        assert len(calls) == len(r['calls']) and set(calls) == set(z.files)
        for c in r['calls']:
            assert c['status'] == 'returned'
            assert c['native_layer_replay_calls'] == c['public_FA_calls'] == 36
            v = c['validation']
            assert v['all_passed'] and v['predicates'] == 828 and v['statistics'] == 36
            if not c['name'].startswith('warm/'):
                assert c['compiler_before'] == c['compiler_after']
        rows = []
        for case in r['cases']:
            key = f"{case['dataset']}_{case['index']}"
            source = reference[key]
            for field in ('input_sha256', 'prompt_length', 'target_length'):
                assert case[field] == source[field]
            assert sha(np.asarray(source['input_ids'], dtype=np.int64).tobytes()) == case['input_sha256']
            names = [f'{p}/{m}/{key}' for p in ('warm', 'r0', 'r1') for m in ('baseline', 'candidate')]
            assert all(z[n].shape == (len(source['input_ids']),) and np.isfinite(z[n]).all() for n in names)
            assert all(np.array_equal(z[names[0]], z[n]) for n in names)
            row = {'case': key, 'length': len(source['input_ids']), 'all_six_complete_vectors_equal': True}
            for m in ('baseline', 'candidate'):
                measured = [calls[f'{p}/{m}/{key}'] for p in ('r0', 'r1')]
                row[m + '_seconds'] = statistics.mean(c['seconds'] for c in measured)
                row[m + '_peak_bytes'] = max(c['peak_allocated'] for c in measured)
                row[m + '_first_shape_seconds'] = calls[f'warm/{m}/{key}']['seconds']
            row['reduction_fraction'] = 1 - row['candidate_seconds'] / row['baseline_seconds']
            rows.append(row)
        total = lambda m: sum(c['seconds'] for c in r['calls'] if c['name'].startswith(('r0/' + m, 'r1/' + m))) / 2
        for m in ('baseline', 'candidate'):
            assert total(m) == r[m + '_seconds_per_pass']
        reduction = 1 - total('candidate') / total('baseline')
        assert reduction == r['warm_reduction_fraction']
        assert r['efficiency_gate_passed'] == (reduction >= .03)
        summary = {'status': 'verified', 'result_sha256': sha((path / 'results.json').read_bytes()),
                   'vectors_sha256': r['vectors_sha256'], 'cases': count, 'complete_calls': 6 * count,
                   'sample_batch': 1, 'endpoint_batch': 2, 'baseline_seconds': total('baseline'),
                   'candidate_seconds': total('candidate'), 'reduction_fraction': reduction,
                   'efficiency_gate_passed': r['efficiency_gate_passed'],
                   'warm_peak_bytes': {m: max(row[m + '_peak_bytes'] for row in rows) for m in ('baseline', 'candidate')},
                   'rows': rows,
                   'quality': 'All six complete vectors equal within each sample/process: signed RISE ordering and positive MAS/needle inputs unchanged. No new score calls or claim of quality gain.',
                   'timing': 'Two interleaved complete warm rounds. First-shape costs share compiler caches and are not independent cold-start comparisons. Baseline already includes adopted deferred scheduling.'}
        (path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', newline='\n')
        with (path / 'costs.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summaries[job] = {k: v for k, v in summary.items() if k != 'rows'}
    (HERE / 'qwen3_retained_summary.json').write_text(json.dumps(summaries, indent=2) + '\n', newline='\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
