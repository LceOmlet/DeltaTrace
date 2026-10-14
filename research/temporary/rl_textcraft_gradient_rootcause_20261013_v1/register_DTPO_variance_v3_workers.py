"""Record observed v3 initial native states and existing passive recorder."""
import hashlib
import json
from pathlib import Path
import time
import psutil
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job=json.loads((OUT/'DTPO-variance-v3-job.json').read_bytes());run=Path(job['run'])
assert psutil.Process(job['pid']).create_time()==job['birth']
receipt=run/'runtime-overrides/initial-state-and-existing-passive-records-v3.json'
state=json.loads(receipt.read_bytes());assert state['complete'] and state['initial_optimizer0_scheduler0_verified']
source=json.loads((run/'source.json').read_bytes())
source.update(actual_initial_worker_state_verified=True,actual_initial_worker_state_receipt=str(receipt),
    actual_initial_worker_states=state['workers'],passive_credit_recording_runtime_override=str(receipt),
    passive_recording_status='Original passive recorder mounted; first actual DT output pending.',
    actual_import_scope='Actual initial actor/core/worker/checkpoint imports and empty optimizer/scheduler read through original RPC.',
    runtime_verification_status='initial_native_workers_verified_first_complete_iteration_pending')
source['deployment'].update(status='v3_actual_initial_workers_verified',rootcause_repair_verified=False)
(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
for name in ('formal-training.json','active-training.json','active-source.json'):
    path=ROOT/name;manifest=json.loads(path.read_bytes())
    matches=[j for j in manifest['jobs'] if j.get('pid')==job['pid']];assert len(matches)==1
    j=matches[0];others=[x.copy() for x in manifest['jobs'] if x is not j]
    j.update(status='v3_actual_initial_workers_verified_first_complete_iteration_pending',
        actual_initial_worker_state_verified=True,actual_initial_worker_state_receipt=str(receipt),
        workers=state['workers'],passive_credit_recording_runtime_override=str(receipt),
        passive_recording_status=source['passive_recording_status'])
    assert [x for x in manifest['jobs'] if x is not j]==others
    path.write_text(json.dumps(manifest,indent=2)+'\n')
job['status']='v3_actual_initial_workers_verified_first_complete_iteration_pending'
job['actual_initial_worker_state_receipt']=str(receipt)
(OUT/'DTPO-variance-v3-job.json').write_text(json.dumps(job,indent=2)+'\n')
(run/'job.json').write_bytes((OUT/'DTPO-variance-v3-job.json').read_bytes())
report=dict(unix=time.time(),job=job,initial_state_receipt=str(receipt),
    initial_state_SHA256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    configuration=[w['configuration'] for w in state['workers']],passive_records=state['passive_records'],
    rootcause_repair_verified=False)
dest=run/'manifest-registration-v3.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
