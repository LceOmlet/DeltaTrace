"""Diagnostic callback invoked through VERL only after the replay returns."""
from pathlib import Path
import hashlib
import json
import runpy


SOURCE = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-incident37-after-native-report-CPU-source-20261011-v1/source/after_native_report_cpu_source.py')
SOURCE_SHA = '251ca219fba4c1479dda275e633a2a56971b89f9d1a67928848df2af306e7759'


def read_CPU_source(worker, output):
    # No model/FSDP transition or GPU read. Reuse the existing CPU observer;
    # load it here, after the original worker method has returned or raised.
    output = Path(output)
    terminal = output / f'rank{worker.rank}.json'
    if not terminal.exists():
        return {'rank': worker.rank, 'captured': False,
                'reason': 'No original terminal report; observation unavailable.'}
    report = json.loads(terminal.read_bytes())
    if report.get('phase') not in ('failed', 'complete'):
        return {'rank': worker.rank, 'captured': False,
                'reason': 'Original replay is not terminal.'}
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    namespace = runpy.run_path(str(SOURCE))
    namespace['capture'](worker, output / f'rank{worker.rank}-native-GEMM-capture.json')
    path = output / f'rank{worker.rank}-after-native-report-CPU-source.json'
    return {'rank': worker.rank, 'captured': True, 'original_phase': report['phase'],
            'report': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
