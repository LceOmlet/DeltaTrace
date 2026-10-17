"""Reuse existing physical dmon/phase reduction with fresh campaign identities."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/summarize_native_utilization_v2.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e6f3b17198f34ca0b5946637dcbd3ee4920bb08b9783bbc374ffe85bb14199e3'
state = json.loads((ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v2/campaign-state.json').read_bytes())
first_driver_birth = state['jobs'][0]['birth']
code = source.read_text().replace("AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'",
    "AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'")
# The physical producer remains the same independent, existing CPU observer.
code = code.replace("PHYSICAL = AUDIT / 'native-mx-smi-dmon-timestamped.jsonl'",
    "PHYSICAL = ROOT / 'receipts/six-run-native-utilization-20261017-v1/native-mx-smi-dmon-timestamped.jsonl'")
code = code.replace('-8gpu-200-20261017-v1', '-8gpu-200-20261017-v2')
assert code.count('lower, upper = None, None') == 1
code = code.replace('lower, upper = None, None', f'lower, upper = {first_driver_birth!r}, None')
exec(compile(code, str(source), 'exec'), globals())
