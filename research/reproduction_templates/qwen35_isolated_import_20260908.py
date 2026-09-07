"""Install pinned public packages in an isolated environment and check real imports."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = Path('${PRIVATE_MOUNT_PATH}')
PINNED = {
    ROOT / 'wheels/tf513_full/transformers-5.13.0-py3-none-any.whl': '8adbc1d20bd5463cd6876b2eb7cb31971e1065788e7dc6bc12bab597a7c504b7',
    ROOT / 'wheels/veomni/flash_linear_attention-0.4.1-py3-none-any.whl': 'd18bdfe9d1f4b424676444eac9d50fb8433b70e5d4e0e0878b20bcbcdbea57ce',
    ROOT / 'wheels/veomni/fla_core-0.4.1-py3-none-any.whl': '93c6afe4c80fc7bc705fa8aeea6a46d2cf2d77383f9619a41863c7114c801bab',
}
r = {'status': 'running', 'steps': [], 'model_loads': 0, 'model_forwards': 0,
     'study_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'selection': 'FLA0.4.1 has no Triton>=3.3 metadata requirement;0.5.1 CUDA extra does. This choice does not assert0.4.1 runtime compatibility.',
     'base_environment_modified': False}


def save():
    tmp = A / 'results.partial'
    tmp.write_text(json.dumps(r, indent=2))
    tmp.replace(A / 'results.json')


def run(label, command, timeout):
    start = time.time()
    log = A / (label + '.log')
    with log.open('w') as f:
        proc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                              env={**os.environ, 'MACA_PATH': '/opt/maca', 'HF_HUB_OFFLINE': '1',
                                   'PYTHONDONTWRITEBYTECODE': '1'})
    r['steps'].append({'stage': label, 'exit_code': proc.returncode,
                       'seconds': time.time() - start, 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    save()
    if proc.returncode:
        raise RuntimeError(label + ' failed; retained complete log')


save()
try:
    for path, expected in PINNED.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    run('venv', [sys.executable, '-m', 'venv', '--system-site-packages', str(A / 'env')], 60)
    python = str(A / 'env/bin/python')
    run('install', [python, '-m', 'pip', 'install', '--no-index', '--disable-pip-version-check',
                   '--find-links', str(ROOT / 'wheels/tf513_full'),
                   '--find-links', str(ROOT / 'wheels/veomni'), *[str(p) for p in PINNED]], 120)
    probe = '''import json,inspect,importlib.metadata
import torch,triton,transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
assert m.chunk_gated_delta_rule is chunk_gated_delta_rule
assert m.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
assert m.is_fast_path_available
print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,'transformers':transformers.__version__,
 'model_source':inspect.getsourcefile(m),'chunk_source':inspect.getsourcefile(chunk_gated_delta_rule),
 'recurrent_source':inspect.getsourcefile(fused_recurrent_gated_delta_rule),'fast_path_available':m.is_fast_path_available}))
'''
    (A / 'import_probe.py').write_text(probe)
    run('import', [python, str(A / 'import_probe.py')], 90)
    r['status'] = 'real_FLA_imports_available_GPU_dispatch_not_validated'
except Exception:
    r['status'] = 'isolated_preflight_failed_no_model_load'
    r['error'] = traceback.format_exc()
finally:
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [Path(__file__), A / 'results.json', *A.glob('*.log'), *A.glob('import_probe.py')]:
            z.write(p, p.name)
    print(json.dumps({'status': r['status'], 'steps': r['steps']}), flush=True)
