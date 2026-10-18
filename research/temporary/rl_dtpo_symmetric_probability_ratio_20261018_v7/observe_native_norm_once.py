"""Bind the existing one-shot native metadata observer to this eight-rank run.

Only run identity and the already recorded DP dimensions differ from the
existing observer. Its original update, optimizer, return objects and cleanup
are preserved. No new gradient computation, normalization or logger is added.
"""
import hashlib
import json
from pathlib import Path
import sys

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
RUN = ROOT / 'runs/textcraft-dtpo-symmetric-probability-ratio-v7-8gpu-200-20261018-v7'
DEST = AUDIT / 'native-norm-once-v7-v1'
OWNER = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/install_native_rank_metadata_once_v76.py'
OWNER_SHA = '91656ff83424558a4e5d425fb344b1cb3b22e4e7d3dc25ba19ba88369f79dd31'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA

job = json.loads((RUN / 'launch.json').read_bytes())
assert Path(job['run']) == RUN and job['method'] == 'dt'
assert psutil.Process(job['pid']).create_time() == job['birth']
bindings = json.loads((RUN / 'runtime-overrides/actual-worker-imports-after-first-update-v1.json').read_bytes())
workers = {r['pid']: r['birth'] for r in bindings['results']}
assert len(workers) == 8
assert all(r['DP_world_size'] == 8 and r['worker_minibatch'] == 8 and r['microbatch_per_gpu'] == 4
           and r['DT_CREDIT_MODE'] == 'reward_target_symmetric_probability_ratio_v7'
           for r in bindings['results'])

source = OWNER.read_text()
substitutions = {
    "OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'": f'OUT = Path({str(AUDIT)!r})',
    "RUN = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'": f'RUN = Path({str(RUN)!r})',
    "DEST = OUT / 'live-native-rank-metadata-once-v76'": f'DEST = Path({str(DEST)!r})',
    "JOB = OUT / 'DTPO-variance-v3-job.json'": "JOB = DEST / 'job.json'",
    'WORKERS = {3151306: 1791993626.33, 3153050: 1791993643.98}': f'WORKERS = {workers!r}',
    'w.world_size == 2': 'w.world_size == 8',
    'actor.config.ppo_mini_batch_size == 32': 'actor.config.ppo_mini_batch_size == 8',
    'assert len(names) == 2': 'assert len(names) == 8',
    'Original token-mean B4 accumulation and native two-rank reduction.': 'Original token-mean B4 accumulation and native eight-rank reduction.',
    'native-rank-metadata-once-v76.json': 'native-rank-metadata-once-symmetric-v7-v1.json',
    'Retain both original rank metadata dictionaries': 'Retain all eight original rank metadata dictionaries',
}
for before, after in substitutions.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
code = compile(source, str(OWNER), 'exec')

if '--check-only' in sys.argv:
    print(json.dumps(dict(status='original_observer_identity_and_DP_binding_checked',
        owner=str(OWNER), owner_SHA256=OWNER_SHA, driver_pid=job['pid'], driver_birth=job['birth'],
        workers=workers, substitutions=substitutions, original_update_and_cleanup_body_preserved=True,
        model_calls=0, backward_calls=0, optimizer_updates=0)))
    raise SystemExit(0)

DEST.mkdir(exist_ok=True)
job_path = DEST / 'job.json'
if job_path.exists():
    saved = json.loads(job_path.read_bytes())
    assert (saved['pid'], saved['birth'], saved['run']) == (job['pid'], job['birth'], job['run'])
else:
    job_path.write_text(json.dumps(job, indent=2) + '\n')
exec(code, dict(__name__='__main__', __file__=str(Path(__file__).resolve())))
