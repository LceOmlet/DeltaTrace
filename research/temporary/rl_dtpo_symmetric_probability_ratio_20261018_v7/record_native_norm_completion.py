"""Bind the existing native norm receipt accounting to the eight-rank v7 run."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OWNER = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/record_native_rank_metadata_completion_v76.py'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == '2360a663a646e3c56d8c7e3f144cf4943a6841707f55bd46dfe6ab4cd1733bfc'
source = OWNER.read_text()
changes = {
    "OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'": "OUT = ROOT / 'receipts/dtpo-symmetric-probability-ratio-20261018-v7'",
    "RUN = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'": "RUN = ROOT / 'runs/textcraft-dtpo-symmetric-probability-ratio-v7-8gpu-200-20261018-v7'",
    "directory = OUT / 'live-native-rank-metadata-once-v76'": "directory = OUT / 'native-norm-once-v7-v1'",
    'for rank in (0, 1):': 'for rank in range(8):',
    "value['completed_iteration_before_update'] == 5": "value['completed_iteration_before_update'] == 2",
    'global_per_step = [math.hypot(a, b) for a, b in zip(*partial)]': 'global_per_step = [math.hypot(*items) for items in zip(*partial)]',
    'one_natural_iteration6_native_metadata_observation_completed_methods_restored': 'one_natural_iteration3_native_metadata_observation_completed_methods_restored',
    'sqrt(rank0_partial_norm**2 + rank1_partial_norm**2); then average.': 'sqrt(sum(rank_partial_norm**2 for all eight ranks)); then average.',
    'original_methods_restored_both_ranks=True': 'original_methods_restored_all_eight_ranks=True',
    'native-rank-metadata-once-v76-completed.json': 'native-rank-metadata-once-symmetric-v7-v1-completed.json',
    'completed_iteration=6': 'completed_iteration=3',
    "print(json.dumps(dict(receipt=str(path), report=result)))": "print(json.dumps(dict(receipt=str(path), status=result['status'], native_local_norm_mean_by_rank=result['native_local_norm_mean_by_rank'], materialized_global_norm_per_optimizer_step=result['materialized_global_norm_per_optimizer_step'], materialized_global_norm_mean=result['materialized_global_norm_mean'], original_methods_restored_all_eight_ranks=True)))",
}
for before, after in changes.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(OWNER), 'exec'), globals())
