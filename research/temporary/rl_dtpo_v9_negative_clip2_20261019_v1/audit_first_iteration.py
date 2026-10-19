"""Reuse the original natural-iteration audit, with clip1 and v8 references."""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1/audit_first_iteration.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='b5c0c314d267485adf4d30fc17c27a2c9ff14b9c617e76863750c14ebc314f42'
sys.path.insert(0,str(owner.parent))
code=owner.read_text(encoding='utf-8')
# Retain the complete original log/readout/metadata audit. The old prose below
# this boundary contains fixed historical numbers and is not reused.
code=code[:code.index("reference = report['references'][1]")]
code=code.replace('dtpo-v9-clip-u-postgroup-variance-20261019-v2',
    'dtpo-v9-negative-clip2-postgroup-variance-20261019-v4')
code=code.replace('textcraft-eight-gpu-v9-clip-u-20261019-v2',
    'textcraft-eight-gpu-v9-negative-clip2-20261019-v4')
code=code.replace("('previous_v6_eight_GPU_first_iteration','textcraft-eight-gpu-sequential-20261018-v5')",
    "('previous_clip1_B4_release_first_iteration','textcraft-eight-gpu-v9-clip-u-20261019-v2')")
code=code.replace('efficiency_equivalent_to_previous_eight_GPU_first_iteration=False',
    'efficiency_equivalent_to_previous_eight_GPU_first_iteration=None')
anchor="assert len(readouts)==8, 'Audit exactly the first natural eight-rank DT call'"
assert code.count(anchor)==1
code=code.replace(anchor,"""first_by_worker_log={}
for readout in readouts:first_by_worker_log.setdefault(readout['path'],readout)
readouts=list(first_by_worker_log.values())
raw=[r for r in raw if r['path'] in first_by_worker_log and r['line']<=first_by_worker_log[r['path']]['line']]
"""+anchor)
code=code.replace('Previous v6/v8 are separate methods/runs.',
    'Previous clip1/v8 are separate methods/runs; compare actual event rows and tokens.')
code=code.replace('results_dtpo_v9_first_iteration_efficiency_20261019_v2.json',
    'results_dtpo_v9_negative2_first_iteration_efficiency_20261019_v4.json')
exec(compile(code,str(owner),'exec'),globals())
