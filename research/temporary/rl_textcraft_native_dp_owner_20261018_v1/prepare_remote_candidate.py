"""Stage only the separately versioned, still unaccepted native owner candidate."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-native-dp-owner-20261018-v1'
CANDIDATE = ROOT / 'candidates/textcraft-native-dp-owner-20261018-v1'
BASE = ROOT / 'candidates/textcraft-eight-gpu-sequential-20261017-v1/verl-owner'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert sha(BASE / 'verl/workers/fsdp_workers.py') == '198f509c9a0af8a03d54f22e66f5f0d5593ab4c9eb7b27777c201ef323d2fecb'
source = json.loads((AUDIT / 'receipts/source-preparation-v2.json').read_bytes())
assert not (AUDIT / 'native-benchmark/job.json').exists()
CANDIDATE.mkdir(exist_ok=True)
if not (CANDIDATE / 'verl-owner').exists():
    subprocess.run(['cp', '-a', str(BASE), str(CANDIDATE / 'verl-owner')], check=True)
entry = CANDIDATE / 'entry'
entry.mkdir(exist_ok=True)
for name in ('textcraft_owner_rollout.py', 'textcraft_native_dp_owner.py',
             'owner_environment_configs.json'):
    path = AUDIT / 'source' / name
    assert sha(path) == source['source_SHA256'][name]
    shutil.copy2(path, entry / name)
path = AUDIT / 'source/fsdp_workers.py'
assert sha(path) == source['source_SHA256']['fsdp_workers.py']
shutil.copy2(path, CANDIDATE / 'verl-owner/verl/workers/fsdp_workers.py')
unchanged = {
    'verl/workers/actor/dp_actor.py': '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'verl/trainer/ppo/core_algos.py': 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'verl/workers/sharding_manager/fsdp_vllm.py': '364e69455cff871d3dd842063f216e882a3eccbdd38322fa2bd731b6e5f0bcc2',
    'verl/utils/checkpoint/fsdp_checkpoint_manager.py': 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
}
for name, expected in unchanged.items():
    assert sha(BASE / name) == sha(CANDIDATE / 'verl-owner' / name) == expected
record = dict(status='prepared_only_unaccepted_no_training_launch_or_default_path_change',
    candidate=str(CANDIDATE), baseline=str(BASE), numerical_owners_unchanged=unchanged,
    sources={str(p): sha(p) for p in entry.iterdir() if p.is_file()},
    worker_SHA256=sha(CANDIDATE / 'verl-owner/verl/workers/fsdp_workers.py'))
(AUDIT / 'receipts/remote-candidate-prepared-v2.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
