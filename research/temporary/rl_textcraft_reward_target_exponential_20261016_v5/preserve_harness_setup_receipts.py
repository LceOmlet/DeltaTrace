"""Preserve only the test hook receipts accidentally placed in the future run."""
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
launch=json.loads((AUDIT/'v5-prepared/launch.json').read_bytes())
run=Path(launch['run'])
assert launch['status']=='prepared_only_not_deployed'
assert run==ROOT/'runs/textcraft-dt-reward-target-exponential-20261016-v5'
assert not (run/'launch.json').exists() and not (run/'job.json').exists()
if run.exists():
    assert {p.name for p in run.iterdir()}=={'worker-runtime-overrides'}
    (run/'worker-runtime-overrides').rename(AUDIT/'failed-harness-import-r1/worker-runtime-overrides')
    run.rmdir()  # Empty directory only; every receipt was preserved above.
print('Preserved test-only setup receipts outside the unsubmitted formal run.')
