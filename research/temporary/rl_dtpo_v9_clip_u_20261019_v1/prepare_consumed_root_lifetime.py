"""Prepare one isolated resource-only owner patch; never import or deploy it.

Free the already-consumed replay checkpoints independently of whether Torch's
inactive pinned cache is emptied. The current flag=1 path and observer path
keep their original behavior; flag=0 may reuse inactive owner buffers earlier.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'dt-native-consumed-root-lifetime-20261019-v1'
BASE = ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace'
CANDIDATE = ROOT / 'candidates' / VERSION / 'deltatrace'
AUDIT = ROOT / 'receipts' / VERSION
RELATIVE = Path('clean/qwen35/qwen35_dense_finite_runner.py')
BASE_SHA = '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(BASE / RELATIVE) == BASE_SHA
assert not CANDIDATE.exists(), 'Preserve an existing resource candidate'
original = (BASE / RELATIVE).read_text()
blocks = [
    (
        "        release_replay_host=(observer is None and self.offload_replay_mixer\n"
        "            and os.environ.get('DT_NATIVE_HOST_CACHE_REPLAY_LAYER')=='1')\n",
        "        release_consumed_host=observer is None and self.offload_replay_mixer\n"
        "        release_replay_host=(release_consumed_host\n"
        "            and os.environ.get('DT_NATIVE_HOST_CACHE_REPLAY_LAYER')=='1')\n",
    ),
    (
        "            if release_replay_host:\n"
        "                # expected consumed this endpoint above; later reverse layers\n"
        "                # read only smaller indices. root['0'] survives for signed.\n"
        "                del root['final_norm_input' if i==31 else str(i+1)]\n"
        "                del kwargs[str(i)]\n"
        "                torch._C._host_emptyCache()\n",
        "            if release_consumed_host:\n"
        "                # expected consumed this endpoint above; later reverse layers\n"
        "                # read only smaller indices. root['0'] survives for signed.\n"
        "                del root['final_norm_input' if i==31 else str(i+1)]\n"
        "                del kwargs[str(i)]\n"
        "                if release_replay_host:torch._C._host_emptyCache()\n",
    ),
]
modified = original
for old, new in blocks:
    assert modified.count(old) == 1
    modified = modified.replace(old, new, 1)
ast.parse(modified)
restored = modified
for old, new in reversed(blocks):
    assert restored.count(new) == 1
    restored = restored.replace(new, old, 1)
assert restored == original
# File-level preparation and exact inverse check are not native validation.
shutil.copytree(BASE, CANDIDATE, symlinks=True,
    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
(CANDIDATE / RELATIVE).write_text(modified, newline='\n')
changed = [str(p.relative_to(BASE)) for p in BASE.rglob('*.py')
    if '__pycache__' not in p.parts and sha(p) != sha(CANDIDATE / p.relative_to(BASE))]
assert changed == [str(RELATIVE)]
AUDIT.mkdir(parents=True, exist_ok=True)
(AUDIT / 'resource-only.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), modified.splitlines(True),
    fromfile=str(BASE / RELATIVE), tofile=str(CANDIDATE / RELATIVE))), newline='\n')
receipt = dict(status='prepared_only_not_natively_validated_not_deployed',
    collected=time.time(), version=VERSION, original_root=str(BASE), candidate_root=str(CANDIDATE),
    changed_sources=changed, original_runner_SHA256=BASE_SHA,
    candidate_runner_SHA256=sha(CANDIDATE / RELATIVE),
    generator_SHA256=sha(__file__), original_statements_restored_byte_exact=True,
    observed_owner_coupling='Current flag=0 retains all root/kwargs checkpoints until the complete B4 call returns, despite their accepted flag=1 last-reader boundaries.',
    scope='Separate consumed tensor lifetime from inactive-cache eviction; native Torch allocator remains the sole cache owner.',
    observer_behavior_unchanged=True, flag1_behavior_unchanged=True,
    root0_retained_until_original_final_token_effect=True,
    numerical_performance_and_capacity_verification='pending same-state native DT comparison; preparation is not acceptance',
    formal_run_mutations=0, additional_model_calls=0, additional_DT_calls=0, optimizer_updates=0)
(AUDIT / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
