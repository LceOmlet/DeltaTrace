"""Record completed behavior checks and exact isolated source bindings."""
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

suite = next(ET.parse(AUDIT/'CPU-behavior-tests.xml').getroot().iter('testsuite'))
assert suite.attrib['tests'] == '41'
assert all(suite.attrib[k] == '0' for k in ('errors', 'failures', 'skipped'))
sources = {}
for task in ('textcraft', 'appworld'):
    path = AUDIT/(task+'-source-prepared.json')
    prepared = json.loads(path.read_bytes())
    for file, expected in prepared['source_bindings'].items():
        assert sha(file) == expected, file
    sources[task] = dict(receipt=str(path), SHA256=sha(path),
        changes=prepared['changes'], preserved_owners=prepared['preserved_owners'])
record = dict(status='passed', recorded=time.time(), tests=int(suite.attrib['tests']),
    test_timestamp=suite.attrib['timestamp'], test_seconds=float(suite.attrib['time']),
    suite=str(AUDIT/'source/test_symmetric_credit.py'),
    suite_SHA256=sha(AUDIT/'source/test_symmetric_credit.py'),
    log=str(AUDIT/'CPU-behavior-tests.log'), log_SHA256=sha(AUDIT/'CPU-behavior-tests.log'),
    xml=str(AUDIT/'CPU-behavior-tests.xml'), xml_SHA256=sha(AUDIT/'CPU-behavior-tests.xml'),
    sources=sources, native_cuda_initialized=False,
    checks=['signed symmetric transform and extremes', 'per-event native DP transport and sum',
        'self/future/observation/zero-credit masks', 'no pre- or post-multiplier credit normalization',
        'unchanged native trainer advantage/return handoff', 'v6 and default regression suite',
        '32768 CPU tensor extent only'],
    GPU_capacity_or_training_quality_claim=False, extra_rollouts=0, optimizer_steps=0)
(AUDIT/'CPU-behavior-verification.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(status=record['status'], tests=record['tests'],
    receipt=str(AUDIT/'CPU-behavior-verification.json'))))
