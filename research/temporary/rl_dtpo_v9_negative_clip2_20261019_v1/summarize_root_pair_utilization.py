"""Reduce existing physical samples within the completed same-input DT pair.

Reuse the original dmon/adjacent-phase reducer without another parser. Only
the shared interval in which all eight ranks execute the same side is used.
These are validation intervals, not production iteration speed measurements.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/summarize_native_utilization_v2.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e6f3b17198f34ca0b5946637dcbd3ee4920bb08b9783bbc374ffe85bb14199e3'
paired = json.loads((AUDIT / 'existing-DT-root-pageable-pair-v2-results.json').read_bytes())
rows = [r['value'] for r in paired['native_timing_records'] if r['value']['minibatch'] == 5]
assert len(rows) == 8 and {r['rank'] for r in rows} == set(range(8))
assert all(r['optional_same_input_pageable_pair']['status'] == 'same_input_owner_pair_completed_not_deployed' for r in rows)
windows = {
    'baseline': (max(r['started'] for r in rows), min(r['finished'] for r in rows)),
    'candidate': (max(r['optional_same_input_pageable_pair']['candidate_started_unix'] for r in rows),
                  min(r['optional_same_input_pageable_pair']['candidate_finished_unix'] for r in rows)),
}
reports = {}
for side, (lower, upper) in windows.items():
    destination = AUDIT / ('root-only-pair-' + side + '-utilization')
    destination.mkdir(exist_ok=True)
    code = source.read_text()
    code = code.replace("AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'",
                        'AUDIT = Path(' + repr(str(destination)) + ')')
    code = code.replace("PHYSICAL = AUDIT / 'native-mx-smi-dmon-timestamped.jsonl'",
                        "PHYSICAL = ROOT / 'receipts/six-run-native-utilization-20261017-v1/native-mx-smi-dmon-timestamped.jsonl'")
    code = code.replace('-8gpu-200-20261017-v1', '-8gpu-200-20261019-v4')
    assert code.count('lower, upper = None, None') == 1
    code = code.replace('lower, upper = None, None', f'lower, upper = {lower!r}, {upper!r}')
    namespace = {'__name__': 'original_utilization_reducer_' + side, '__file__': str(source)}
    exec(compile(code, str(source), 'exec'), namespace)
    namespace['main']()
    reports[side] = json.loads((destination / 'native-utilization-phase-summary.json').read_bytes())
result = dict(status='completed_read_only_same_input_pair_utilization',
              source=str(source), source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
              pair_receipt=str(AUDIT / 'existing-DT-root-pageable-pair-v2-results.json'),
              reports=reports, training_mutations=0, additional_model_calls=0,
              scope='Validation-only paired natural B4; exclude from production iteration speed claims. Both adjacent observations must agree for all eight ranks; missing/mixed brackets remain separate.')
(AUDIT / 'root-only-pair-utilization.json').write_text(json.dumps(result, indent=2) + '\n')
