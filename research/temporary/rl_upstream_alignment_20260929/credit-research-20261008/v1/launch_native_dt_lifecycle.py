"""Launch the isolated, bounded native DT lifecycle diagnostic; no training launch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


task, directory = sys.argv[1:]
out = Path(directory)
root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
assert task in ('textcraft', 'appworld')
assert not (out/'launch.json').exists(), 'Do not duplicate this diagnostic'
hashes = json.loads((out/'native-dt-lifecycle-source-hashes.json').read_bytes())['files']
for name, digest in hashes.items():
    assert sha(out/name) == digest
    if name.endswith('.py'):
        ast.parse((out/name).read_text())
source_path = root/'runs/direct-target-prefix-runtime-20261007-v1'/task/(task+'-dt')/'source.json'
expected = dict(textcraft='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',
    appworld='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0')
assert sha(source_path) == expected[task]
source = json.loads(source_path.read_bytes())
physical = subprocess.run(['mx-smi'], capture_output=True, text=True, check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S', physical, re.M), 'Research devices occupied'
(out/'before-physical.txt').write_text(physical)
env = dict(os.environ, **source['environment'])
env.pop('MACA_VISIBLE_DEVICES', None)
env.pop('RAY_ADDRESS', None)
env['CUDA_VISIBLE_DEVICES'] = '4,5'
env['DT_TASK'] = source['startup_options']['env.env_name']
env['DT_MAX_STEPS'] = str(source['startup_options']['env.max_steps'])
dt = Path(source['dt_root'])
qwen = json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official = env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
env['PYTHONPATH'] = ':'.join([str(out), str(dt), official, str(dt/'clean/qwen35'),
    source['pythonpath'], qwen['ft_extension_root']])
argv = [env['VENV_PYTHON'], str(out/'inspect_native_dt_lifecycle.py'),
    '--source', str(source_path), '--output', str(out/'results'), '--case', task]
if task == 'textcraft':
    evidence = root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
    assert sha(evidence) == '57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
    argv += ['--owner-total-training-steps', '330', '--owner-total-steps-evidence', str(evidence)]
with (out/'driver.log').open('xb') as stream:
    process = subprocess.Popen(argv, env=env, cwd=out, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(task=task, pid=process.pid, birth=psutil.Process(process.pid).create_time(),
    launched_unix=time.time(), devices=[4, 5], base_commit='1a7676c6',
    uncommitted_diagnostic_files_bound_by_sha256=hashes, argv=argv,
    source_path=str(source_path), source_sha256=sha(source_path),
    model_precision_and_training_config='unchanged; native before/after DT and original observer isolation only',
    optimizer=0, expected_DT_B4_per_rank=6, checkpoint_restore=0, production_patches=0,
    wall_budget_per_worker_seconds=1800)
(out/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
