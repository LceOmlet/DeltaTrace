"""Aggregate completed retained timing intervals, without new model calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    path = HERE / 'qwen35_retained16/results.json'
    raw = path.read_bytes()
    report = json.loads(raw)
    manifest = json.loads((ROOT / 'evidence/export_manifest.json').read_bytes())
    entry = next(x for x in manifest['artifacts'] if x['path'] == path.relative_to(ROOT).as_posix())
    assert hashlib.sha256(raw).hexdigest() == entry['public_sha256']
    calls = [c for c in report['calls'] if c['name'].startswith('measured_') and '_accelerated/' in c['name']]
    assert len(calls) == 16 and len({c['name'] for c in calls}) == 16
    stages = {}
    counts = {}
    for call in calls:
        assert call['status'] == 'returned'
        assert call['compiler_before'].get('unique_graphs', 0) == call['compiler_after'].get('unique_graphs', 0)
        assert call['details']['sample_batch'] == 2
        for record in call['details']['calls']:
            key = record['kind']
            group = next((g for g in ('native_root', 'native_replay', 'finite_decoder', 'public_FA_LSE')
                          if key.startswith(g + '_')), key)
            stages[group] = stages.get(group, 0.) + record['stream_elapsed_seconds'] / 2
            counts[group] = counts.get(group, 0) + 1
    assert counts['native_replay'] == counts['finite_decoder'] == 512
    total = sum(c['seconds'] for c in calls) / 2
    summary = {
        'status': 'verified_existing_records',
        'source': path.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'complete_seconds_per16': total,
        'serial_stream_intervals_seconds': stages,
        'residual_to_complete_seconds': total - sum(stages.values()),
        'fractions': {k: v / total for k, v in stages.items()},
        'interval_counts_two_rounds': counts,
        'ideal_speedup_if_finite_decoder_twice_as_fast': total / (total - stages['finite_decoder'] / 2),
        'new_model_calls': 0, 'new_FT_calls': 0, 'new_metric_calls': 0,
        'limits': 'Two interleaved complete retained B2 passes over the original16. '
                  'Serial CUDA event intervals include possible host submission gaps; these '
                  'are not pure-kernel FLOP fractions. The ideal speedup holds all other '
                  'intervals fixed and is not a measured optimization or a comparison to FT.'
    }
    (HERE / 'remaining_speed_space.json').write_text(json.dumps(summary, indent=2) + '\n', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
