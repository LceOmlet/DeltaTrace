"""Link the completed actual v3 input regression to its mutable run receipts.

CPU/JSON bookkeeping only; launch, configuration and training files are not
changed. Other jobs' manifest entries are preserved verbatim as data.
"""
import hashlib
import json
from pathlib import Path
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'live-native-update-input-variance-v3'
job = json.loads((OUT / 'DTPO-variance-v3-job.json').read_bytes())
RUN = Path(job['run'])
assert psutil.Process(job['pid']).create_time() == job['birth']
proof_path = DEST / 'actual-native-input-variance-v3-verification.json'
proof = json.loads(proof_path.read_bytes())
installation_path = RUN / 'runtime-overrides/live-native-update-input-variance-v3.json'
installation = json.loads(installation_path.read_bytes())
assert installation['driver_pid'] == job['pid'] and installation['driver_birth'] == job['birth']
assert proof['integration_regression_passed'] and proof['completed_step_before_update'] == 0
assert proof['rows'] == 256
assert all(all(v.values()) if isinstance(v, dict) else v for v in proof['checks'].values())
assert {c['pid']: c['birth'] for c in proof['captures']} == {
    3151306: 1791993626.33, 3153050: 1791993643.98,
}
assert all(c['original_update_restored'] and c['status'] == 'saved' for c in proof['captures'])
assert proof['model_calls'] == proof['DT_calls'] == proof['optimizer_calls'] == 0
copied = RUN / 'runtime-overrides/actual-native-input-variance-v3-verification.json'
if copied.exists():
    assert copied.read_bytes() == proof_path.read_bytes()
else:
    copied.write_bytes(proof_path.read_bytes())
verification = dict(
    receipt=str(copied), SHA256=hashlib.sha256(copied.read_bytes()).hexdigest(),
    source_receipt=str(proof_path), installation_receipt=str(installation_path),
    completed_step_before_update=0, rows=proof['rows'], checks=proof['checks'],
    variance_scale=proof['variance_scale'],
    raw_DT_masked_std=proof['raw_DT_masked_std'],
    actual_advantage_masked_mean=proof['actual_advantage_masked_mean'],
    actual_advantage_masked_std=proof['actual_advantage_masked_std'],
    scope='Actual first native PPO input boundary verified; not a completed-iteration or gradient-stability claim.',
)
source_path = RUN / 'source.json'
source = json.loads(source_path.read_bytes())
before = DEST / 'source-before-input-registration.json'
if not before.exists():
    before.write_bytes(source_path.read_bytes())
source.update(actual_native_PPO_input_verification=verification,
              runtime_verification_status='v3_actual_first_native_input_boundary_verified')
source['deployment'].update(status='v3_actual_first_native_input_boundary_verified',
                            rootcause_repair_verified=False)
source_path.write_text(json.dumps(source, indent=2) + '\n')
manifest_receipts = {}
for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
    path = ROOT / name
    manifest = json.loads(path.read_bytes())
    matches = [j for j in manifest['jobs'] if j.get('pid') == job['pid']]
    assert len(matches) == 1
    current = matches[0]
    assert current['birth'] == job['birth']
    others = [j.copy() for j in manifest['jobs'] if j is not current]
    backup = DEST / (name + '.before-input-registration')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    current.update(actual_native_PPO_input_verification=verification,
                   status='v3_actual_first_native_input_boundary_verified')
    assert [j for j in manifest['jobs'] if j is not current] == others
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest_receipts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
job.update(actual_native_PPO_input_verification=verification,
           status='v3_actual_first_native_input_boundary_verified')
(OUT / 'DTPO-variance-v3-job.json').write_text(json.dumps(job, indent=2) + '\n')
(RUN / 'job.json').write_bytes((OUT / 'DTPO-variance-v3-job.json').read_bytes())
credit_records = {
    p.name: sum(1 for file in p.rglob('*') if file.is_file())
    for p in (RUN / 'credit-records').iterdir() if p.is_dir()
}
report = dict(unix=time.time(), job={k: job[k] for k in ('pid', 'birth', 'run', 'version')},
              verification=verification, passive_credit_file_counts=credit_records,
              manifest_SHA256=manifest_receipts, rootcause_repair_verified=False,
              training_configuration_changes=0, other_job_changes=0)
path = DEST / 'actual-input-registration.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
