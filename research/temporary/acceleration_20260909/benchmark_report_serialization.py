"""Measure standard-library JSON output formatting on a real saved report.

CPU only. Does not modify an experiment, model, metric or its saved source data.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'complete'
    timings = []
    for repeat in range(2):
        for mode in (('indented', 'compact') if repeat == 0 else ('compact', 'indented')):
            gc.collect()
            started = time.perf_counter()
            kwargs = {'indent': 2} if mode == 'indented' else {'separators': (',', ':')}
            encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, **kwargs).encode('utf-8')
            elapsed = time.perf_counter()-started
            assert json.loads(encoded) == report
            if mode == 'indented':
                assert encoded == raw
            timings.append({'repeat': repeat, 'mode': mode, 'seconds': elapsed,
                            'bytes': len(encoded), 'parsed_content_identical': True,
                            'legacy_bytes_identical': encoded == raw if mode == 'indented' else None})
            del encoded
    result = {'status': 'complete', 'scope': 'Actual complete task report; standard json library serialization only',
              'report_sha256': hashlib.sha256(raw).hexdigest(), 'report_bytes': len(raw),
              'cases': len(report['cases']), 'source_driver_sha256': report['driver_sha256'],
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'timings': timings, 'model_calls': 0, 'metric_calls': 0,
              'experiment_modified': False, 'interpretation': 'A file-output optimization measurement, not attribution latency.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
