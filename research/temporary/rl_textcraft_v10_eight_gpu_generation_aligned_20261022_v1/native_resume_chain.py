"""Compose the pinned native log reader and original same-run resume selection."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PARSER = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
RESUME_OWNER = Path(__file__).with_name('original_approved_DTPO_v63.py')


@lru_cache(maxsize=1)
def original_owners():
    assert hashlib.sha256(PARSER.read_bytes()).hexdigest() == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
    assert hashlib.sha256(RESUME_OWNER.read_bytes()).hexdigest() == 'f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'
    source = RESUME_OWNER.read_text()
    block = source[source.index("run=Path(job['run'])"):
                   source.index("checkpoint=run/'textcraft-ppo/checkpoints/latest_checkpointed_iteration.txt'")]
    return runpy.run_path(str(PARSER), run_name='original_native_resume_log_owner'), block


def collect(root, job_outputs=None):
    owner, block = original_owners()
    snapshot = owner['collect'](root, job_outputs)
    manifest = json.loads((Path(root) / 'formal-training.json').read_text())
    for current in snapshot['jobs']:
        job = next(row for row in manifest['jobs']
                   if row.get('output', row.get('run_dir')) == current['run_dir'])
        if not job.get('previous_job_identity'):
            continue
        # Execute the unchanged native retained-interval selection. All metric
        # parsing and path/line provenance still come from read_job_logs/parse_logs.
        namespace = dict(job=job, Path=Path, owner=owner)
        exec(compile(block, str(RESUME_OWNER), 'exec'), namespace)
        values = namespace['metrics_by_step']
        current.update(metrics=[values[step] for step in sorted(values)],
                       records=namespace['records'], errors=namespace['errors'][-10:],
                       ray_sessions=namespace['sessions'],
                       resumed_completed_iteration=job['resumed_completed_iteration'],
                       resume_checkpoint=job['resume_checkpoint'],
                       native_execution_source_commit=job.get('execution_source_commit'),
                       native_resource_reuse=job.get('runtime_resource_reuse'),
                       resume_scope='Original same-run retained-interval owner; no interpolation or cross-run stitching',
                       previous_failure_errors_retained=True,
                       resume_selection_owner=dict(path=str(RESUME_OWNER),
                           SHA256='f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'))
    return snapshot
