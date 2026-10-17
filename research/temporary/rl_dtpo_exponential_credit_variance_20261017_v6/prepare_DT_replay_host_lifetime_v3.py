"""Prepare an isolated DT resource candidate; do not import, deploy, or run it.

Keep the accepted numerical runner. Release original CPU endpoint checkpoints
only after their last original reverse-pass reader, and call the existing
native inactive-host-cache API at that boundary. The old diagnostic path and
the default-off path keep all original statements.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'dt-native-replay-host-lifetime-20261017-v3'
AUDIT = ROOT / 'receipts' / VERSION
BASE = ROOT / 'candidates/textcraft-dt-reward-target-exponential-20261016-v5/deltatrace'
CANDIDATE = ROOT / 'candidates' / VERSION / 'deltatrace'
RELATIVE = 'clean/qwen35/qwen35_dense_finite_runner.py'
BASE_SHA = '198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

AUDIT.mkdir(parents=True, exist_ok=True)
original = (BASE / RELATIVE).read_text()
assert sha(BASE / RELATIVE) == BASE_SHA
assert not CANDIDATE.exists(), 'Preserve an existing preparation instead of overwriting it'
blocks = [
    ("import torch\n", "import torch\nimport os\n"),
    (
        "        seed_effect=effect(m,x);del x,mnorm,seed,z\n",
        "        seed_effect=effect(m,x);del x,mnorm,seed,z\n"
        "        release_replay_host=(observer is None and self.offload_replay_mixer\n"
        "            and os.environ.get('DT_NATIVE_HOST_CACHE_REPLAY_LAYER')=='1')\n"
        "        if release_replay_host:torch._C._host_emptyCache()\n",
    ),
    (
        "            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])\n",
        "            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])\n"
        "            if release_replay_host:\n"
        "                # expected consumed this endpoint above; later reverse layers\n"
        "                # read only smaller indices. root['0'] survives for signed.\n"
        "                del root['final_norm_input' if i==31 else str(i+1)]\n"
        "                del kwargs[str(i)]\n"
        "                torch._C._host_emptyCache()\n",
    ),
]
modified = original
for old, new in blocks:
    assert modified.count(old) == 1, old
    modified = modified.replace(old, new, 1)
ast.parse(modified)
restored = modified
for old, new in reversed(blocks):
    assert restored.count(new) == 1
    restored = restored.replace(new, old, 1)
assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(
    ast.parse(original), include_attributes=False)
assert restored == original
shutil.copytree(BASE, CANDIDATE, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
(CANDIDATE / RELATIVE).write_text(modified, newline='\n')
(AUDIT / 'original-runner.py').write_text(original, newline='\n')
(AUDIT / 'candidate-runner.py').write_text(modified, newline='\n')
(AUDIT / 'resource-only.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), modified.splitlines(True),
    fromfile=str(BASE / RELATIVE), tofile=str(CANDIDATE / RELATIVE))), newline='\n')
changed = []
for p in BASE.rglob('*.py'):
    if '__pycache__' in p.parts:
        continue
    relative = p.relative_to(BASE)
    if sha(p) != sha(CANDIDATE / relative):
        changed.append(str(relative))
assert changed == [RELATIVE]
receipt = dict(
    status='prepared_resource_candidate_not_deployed_or_validated',
    version=VERSION, collected=time.time(), original_root=str(BASE),
    candidate_root=str(CANDIDATE), changed_sources=changed,
    original_SHA256=BASE_SHA, candidate_SHA256=sha(CANDIDATE / RELATIVE),
    resource_flag='New default-off resource flag DT_NATIVE_HOST_CACHE_REPLAY_LAYER=1; existing B4 reclaim remains enabled; observer must be None and original offload_replay_mixer true',
    original_numerical_statements_recovered_byte_exact=True,
    unchanged_default_and_observer_control_flow='New native reclaim/deletion branches inactive; original numerical statements retained',
    last_consumer=dict(
        released_endpoint="final_norm_input after layer31; root[i+1] after layer i",
        readers='Original expected output comparison/effect and current-layer input retain all needed values',
        root0='Retained until original final _token_effect',
        kwargs='CPU kw checkpoint no longer read after original copied GPU kw is consumed',
    ),
    native_API='torch._C._host_emptyCache; original CachingHostAllocator frees inactive event-ready buffers only',
    measurements_required='Actual native same-input DT vectors and B4/32768 physical host/PSS/XTT/elapsed. Preparation alone proves neither peak reduction nor four-resident safety.',
    formal_training_mutations=0, model_calls=0, optimizer_updates=0,
    full_four_resident_32k_capacity_verified=False,
)
(AUDIT / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
