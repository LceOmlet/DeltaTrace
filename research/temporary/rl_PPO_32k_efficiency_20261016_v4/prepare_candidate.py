"""Keep the verified long-input owner path; retain native HF for proven fitting inputs."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'PPO-32k-native-efficiency-20261016-v4'
BASE = ROOT/'candidates/PPO-32k-native-efficiency-20261016-v2/verl-owner'
DEST = ROOT/'candidates'/VERSION/'verl-owner'
AUDIT = ROOT/'receipts'/VERSION
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(), 'Preserve earlier candidates'
performance = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/status.json'
prior = json.loads(performance.read_bytes())
assert prior['status'] == 'completed' and prior['exact_transport'] == 'passed'
assert prior['comparisons'][0]['context'] == 8192 and prior['comparisons'][0]['median_ratio'] > 1
assert prior['comparisons'][1]['context'] == 32768 and prior['comparisons'][1]['median_ratio'] < 1
assert sha(BASE/'verl/utils/activation_offload.py') == 'f62bd44a203f8ee27480792769281f4259d299beb74db4a215031bd6b6d5b423'
assert sha(BASE/'verl/workers/fsdp_workers.py') == '76f3efbf53e25bd9919ae79de061cd6a47401d8e93df3d5c64bf397e09aa94dd'
shutil.copytree(BASE, DEST, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
AUDIT.mkdir(exist_ok=True)
changes = []
p = DEST/'verl/utils/activation_offload.py'
before = p.read_text()
old = 'def enable_activation_offloading(model, strategy, enable_ckpt=False, preserve_hf_checkpoint=False):'
new = 'def enable_activation_offloading(model, strategy, enable_ckpt=False, preserve_hf_checkpoint=False, native_checkpoint_max_seq_len=0):'
assert before.count(old) == 1
after = before.replace(old, new)
anchor = '            def checkpoint_with_offload(*args, **kwargs):\n                with context:'
replacement = '''            def checkpoint_with_offload(*args, **kwargs):
                # This optional memory-backend bound is supplied only for a
                # measured fitting model/microbatch. HF passes the callable
                # first and the original [batch, sequence, hidden] input next.
                if native_checkpoint_max_seq_len and args[1].shape[1] <= native_checkpoint_max_seq_len:
                    return native_checkpoint(*args, **kwargs)
                with context:'''
assert after.count(anchor) == 1
after = after.replace(anchor, replacement)
assert after.replace(replacement, anchor).replace(new, old) == before
compile(after, str(p), 'exec'); p.write_text(after)
changes.append(dict(path=str(p), before=sha(BASE/'verl/utils/activation_offload.py'), after=sha(p),
                    scope='Default-zero native checkpoint fast path; existing long-input branch byte-for-byte unchanged'))
p = DEST/'verl/workers/fsdp_workers.py'
before = p.read_text()
old = '                    enable_gradient_checkpointing, preserve_hf_checkpoint=True)'
new = '''                    enable_gradient_checkpointing, preserve_hf_checkpoint=True,
                    native_checkpoint_max_seq_len=config.model.get("activation_offload_native_max_seq_len", 0))'''
assert before.count(old) == 1
after = before.replace(old, new)
assert after.replace(new, old) == before
compile(after, str(p), 'exec'); p.write_text(after)
changes.append(dict(path=str(p), before=sha(BASE/'verl/workers/fsdp_workers.py'), after=sha(p),
                    scope='Pass optional memory bound to the existing optional Qwen3.5 FSDP2 critic offload'))
unchanged = {name: sha(DEST/name) for name in (
    'verl/workers/critic/dp_critic.py', 'verl/workers/actor/dp_actor.py',
    'verl/trainer/ppo/core_algos.py', 'verl/utils/checkpoint/fsdp_checkpoint_manager.py',
    'verl/utils/experimental/torch_functional.py')}
assert all(value == sha(BASE/name) for name, value in unchanged.items())
numeric = ROOT/'receipts/PPO-32k-numeric-reference-20261016-v3/same-session-status.json'
assert json.loads(numeric.read_bytes())['numerical_accepted']
record = dict(version=VERSION, status='prepared_only_short_path_unverified', VERL_ROOT=str(DEST),
              base=str(BASE), changes=changes, unchanged_owner_sources=unchanged,
              memory_bound=dict(sequence=8192, batch_per_rank=4, model='Qwen3.5-9B',
                                evidence=str(performance), default=0,
                                meaning='Proven fitting native path; not a numerical tolerance or training hyperparameter'),
              reused_32768=dict(numerical=str(numeric), performance=str(performance),
                               reason='32768 exceeds 8192; unchanged original v2 long-input body and owner calls'),
              formal_training_mutations=0, method_or_training_hyperparameter_changes=[])
(AUDIT/'prepared-candidate.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
