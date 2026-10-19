"""Use the existing strict physical/phase reducer on the three saved calls."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('summarize_root_pair_utilization.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'aae67375d1e443192c140e763404bfc4605bf669b1e4776cba356ae0424aa769'
code = owner.read_text()
for old, new in (
    ('existing-DT-root-pageable-pair-v2-results.json', 'existing-DT-FA-capture-resource-pair-results.json'),
    ('optional_same_input_pageable_pair', 'optional_same_input_FA_capture_pair'),
    ("if r['value']['minibatch'] == 5", "if r['value'].get('optional_same_input_FA_capture_pair')"),
    ('root-only-pair-', 'FA-capture-pair-'),
    ('root-only-pair-utilization.json', 'FA-capture-pair-utilization.json'),
):
    # The file-name replacement above may already replace the final output.
    if old in code:
        code = code.replace(old, new)
anchor = "    'baseline': (max(r['started'] for r in rows), min(r['finished'] for r in rows)),"
assert code.count(anchor) == 1
code = code.replace(anchor, "    'cold_original': (max(r['started'] for r in rows), min(r['finished'] for r in rows)),\n"
    "    'warm_original': (max(r['optional_same_input_FA_capture_pair']['warm_baseline_started_unix'] for r in rows),\n"
    "                      min(r['optional_same_input_FA_capture_pair']['warm_baseline_finished_unix'] for r in rows)),")
exec(compile(code, str(owner), 'exec'), globals())
