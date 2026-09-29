"""Publish the tested source overlay and launch bounded native trainer jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
source = audit / 'distributed-runtime'
release = root / 'releases/fc2e6c2'
assert not release.exists(), 'Release already exists; inspect it rather than overwrite'
shutil.copytree(source, release, ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(audit / 'two-gpu-runtime-release.tar') as archive:
    archive.extractall(release, filter='data')
runner = Path('clean/qwen35/qwen35_dense_finite_runner.py')
shutil.copy2(release / 'deltatrace' / runner, release / runner)
base_receipt = json.loads((release / 'deployment-receipt.json').read_text())
receipt = {
    'revision': 'fc2e6c2', 'purpose': 'Two bounded fresh iterations; not formal training',
    'created_unix': time.time(), 'base_runtime': str(source),
    'base_deployment_receipt': base_receipt,
    'owner': str(root / 'candidates/official-verl-20bd331-distributed-dt'),
    'runtime_sha256': {
        str(path.relative_to(release)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ('clean', 'accelerated', 'profiles', 'experiments/rl')
        for path in (release / directory).rglob('*')
        if path.is_file() and path.suffix in ('.py', '.sh', '.json')
    },
}
(release / 'deployment-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
# Preserve the inherited manifest as history, not as a claim about this overlay.
(release / 'source-manifest.json').rename(release / 'base-source-manifest.json')
script = release / 'research/temporary/rl_upstream_alignment_20260929/run_two_gpu_training_pilot.sh'
subprocess.run(['bash', '-n', str(script)], check=True)
jobs = []
for method, task, devices, offset in (
    ('dt', 'Sokoban', '0,1', None), ('dt', 'Webshop', '2,3', None),
    ('dt', 'AppWorld', '4,5', 0), ('grpo', 'AppWorld', '6,7', 17),
):
    directory = audit / 'two-gpu-pilots' / f'{method}-{task}'
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DT_ROOT=str(release), CUDA_VISIBLE_DEVICES=devices,
               ENV_NAME=task, METHOD=method)
    if offset is not None:
        env['APPWORLD_SERVER_OFFSET'] = str(offset)
    with (directory / 'train.log').open('wb') as log:
        proc = subprocess.Popen(['bash', str(script)], env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    job = dict(method=method, task=task, devices=devices, pid=proc.pid,
               started_unix=time.time(), directory=str(directory), release=str(release),
               budget={'iterations': 2, 'prompt_groups': 4, 'group_size': 4,
                       'max_steps': 15, 'max_response': 512, 'max_total': 32768,
                       'optimizer_minibatch': 64, 'actor_microbatch_per_gpu': 4,
                       'dt_batch_per_gpu': 4},
               status='started_bounded_pilot_not_health_or_formal_acceptance',
               input_length_selection=False)
    (directory / 'job.json').write_text(json.dumps(job, indent=2)+'\n')
    jobs.append(job)
(audit / 'two-gpu-pilots/manifest.json').write_text(json.dumps(jobs, indent=2)+'\n')
print(json.dumps(jobs, indent=2))
