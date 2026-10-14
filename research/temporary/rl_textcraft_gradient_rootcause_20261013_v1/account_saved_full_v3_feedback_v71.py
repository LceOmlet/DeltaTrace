"""Read completed v69 native updates and fixed-seed feedback, CPU only.

Reuse the existing saved-gradient accounting. This does not implement a loss,
optimizer, DT computation, backward pass or a new training experiment.
"""
import hashlib
import json
from pathlib import Path

source = Path(__file__).with_name('account_common_update_feedback_v57.py')
expected = '5e82ac294f7a2aebe890d966adebf5b49fc521e038ddb644c458265375fedcad'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
changes = {
    'Account completed native v47 updates and fixed-seed feedback from saved files.':
        'Account completed full-v3 native updates and fixed-seed feedback from saved files.',
    "DEST=OUT/'native-common-update-feedback-v47'":
        "DEST=OUT/'native-full-current-v3-feedback-v69'",
    'without_constant_positive_component': 'approved_v3_full_method',
    "ablated_growth=ablated['fixed_seed_parameter_VJP_ratio']-1":
        "approved_v3_growth=ablated['fixed_seed_parameter_VJP_ratio']-1",
    'completed-feedback-accounting-v57.json': 'completed-full-current-v3-feedback-accounting-v71.json',
    "print(json.dumps(dict(receipt=str(dest),report=report)))": '',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
preparation_path = DEST / 'full-current-v3-DT-preparation.json'
preparation = json.loads(preparation_path.read_bytes())
assert all(preparation['checks'].values())
report.update(
    accounting_source=dict(path=str(source), SHA256=expected),
    v3_preparation_source=dict(path=str(preparation_path),
        SHA256=hashlib.sha256(preparation_path.read_bytes()).hexdigest()),
    complete_v3_preparation=preparation,
    new_training_calls_in_accounting=0,
    original_native_reduction='Pinned official token-mean, 1/16 sum of B4 means.',
    rootcause_repair_verified=False,
)
report['limits'].append(
    'The second arm is the full approved v3, including new DT requests for '
    'zero-reward/nonzero-group trajectories; it is not the earlier partial-row ablation.')
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))
