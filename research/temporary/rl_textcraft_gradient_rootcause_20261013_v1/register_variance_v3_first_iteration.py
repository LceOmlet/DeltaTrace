"""Register the first completed v3 iteration from the original native parser."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
job_path = OUT / 'DTPO-variance-v3-job.json'
job = json.loads(job_path.read_bytes())
RUN = Path(job['run'])
p = psutil.Process(job['pid'])
assert p.create_time() == job['birth'] and p.status() != psutil.STATUS_ZOMBIE
dest = RUN / 'runtime-overrides/first-completed-native-iteration-v3.json'
assert not dest.exists(), 'Read the completed receipt; do not duplicate it.'
parser_path = OUT / 'original_plot_training_progress.py'
parser = runpy.run_path(str(parser_path), run_name='existing_v3_first_iteration_parser')
sessions = list(Path('/tmp/ray').glob('session_*_' + str(job['pid'])))
assert len(sessions) == 1
records, errors = parser['read_job_logs']({'ray_session': str(sessions[0])})
metrics, phases = parser['parse_logs'](records)
first = next(m for m in metrics if m['step'] == 1)
assert math.isfinite(first['values']['actor/grad_norm'])
input_proof = RUN / 'runtime-overrides/actual-native-input-variance-v3-verification.json'
assert json.loads(input_proof.read_bytes())['integration_regression_passed']
credit_records = {
    d.name: sum(1 for f in d.rglob('*') if f.is_file())
    for d in (RUN / 'credit-records').iterdir() if d.is_dir()
}
assert len(credit_records) == 2 and all(credit_records.values())
report = dict(unix=time.time(), job={k: job[k] for k in ('pid', 'birth', 'run', 'version')},
              first_completed_iteration=first, latest_completed_iteration=max(m['step'] for m in metrics),
              original_parser=dict(path=str(parser_path), SHA256=hashlib.sha256(parser_path.read_bytes()).hexdigest()),
              actual_input_verification=str(input_proof), passive_credit_file_counts=credit_records,
              read_errors=errors, rootcause_repair_verified=False,
              scope='First actual native iteration and input integration observed, not a long-term stability claim.',
              model_calls=0, backward_calls=0, optimizer_calls=0)
dest.write_text(json.dumps(report, indent=2) + '\n')
evidence = dict(receipt=str(dest), SHA256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                observed_unix=report['unix'], completed_iteration=1, rootcause_repair_verified=False)
source_path = RUN / 'source.json'
source = json.loads(source_path.read_bytes())
source.update(first_completed_native_iteration=evidence,
              passive_credit_file_counts=credit_records,
              passive_recording_status='Original passive records have landed on both actual v3 workers.',
              runtime_verification_status='v3_first_complete_native_iteration_and_input_verified')
source['deployment'].update(status='v3_first_complete_native_iteration_and_input_verified',
                            rootcause_repair_verified=False)
source_path.write_text(json.dumps(source, indent=2) + '\n')
for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
    path = ROOT / name
    manifest = json.loads(path.read_bytes())
    matches = [j for j in manifest['jobs'] if j.get('pid') == p.pid]
    assert len(matches) == 1 and matches[0]['birth'] == job['birth']
    current = matches[0]
    others = [j.copy() for j in manifest['jobs'] if j is not current]
    current.update(first_completed_native_iteration=evidence,
                   passive_credit_file_counts=credit_records,
                   passive_recording_status=source['passive_recording_status'],
                   status='v3_first_complete_native_iteration_and_input_verified')
    assert [j for j in manifest['jobs'] if j is not current] == others
    path.write_text(json.dumps(manifest, indent=2) + '\n')
job.update(first_completed_native_iteration=evidence,
           status='v3_first_complete_native_iteration_and_input_verified')
job_path.write_text(json.dumps(job, indent=2) + '\n')
(RUN / 'job.json').write_bytes(job_path.read_bytes())
print(json.dumps(dict(receipt=str(dest), report=report)))
