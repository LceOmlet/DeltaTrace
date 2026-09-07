"""Audit unchanged package sources, apply explicit device mapping, recheck imports."""
import hashlib
import json
import os
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
PARENT = Path('${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1')
ENV = PARENT / 'env'
SITE = ENV / 'lib/python3.12/site-packages'
ROOT = Path('${PRIVATE_MOUNT_PATH}')
sha = lambda b: hashlib.sha256(b).hexdigest()
WHEELS = {
    ROOT / 'wheels/tf513_full/transformers-5.13.0-py3-none-any.whl': '8adbc1d20bd5463cd6876b2eb7cb31971e1065788e7dc6bc12bab597a7c504b7',
    ROOT / 'wheels/veomni/flash_linear_attention-0.4.1-py3-none-any.whl': 'd18bdfe9d1f4b424676444eac9d50fb8433b70e5d4e0e0878b20bcbcdbea57ce',
    ROOT / 'wheels/veomni/fla_core-0.4.1-py3-none-any.whl': '93c6afe4c80fc7bc705fa8aeea6a46d2cf2d77383f9619a41863c7114c801bab',
}
r = {'status': 'running', 'model_loads': 0, 'model_calls': 0, 'kernel_math_changes': 0,
     'study_sha256': sha(Path(__file__).read_bytes()), 'base_environment_modified': False}


def sources(mapped):
    count = 0
    changed = []
    for wheel, expected in WHEELS.items():
        assert sha(wheel.read_bytes()) == expected
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if name.endswith('.py') and name.startswith(('fla/', 'transformers/')):
                    original = z.read(name)
                    actual = (SITE / name).read_bytes()
                    if actual != original:
                        assert mapped and name == 'fla/utils.py'
                        changed.append({'name': name, 'original_sha256': sha(original), 'actual_sha256': sha(actual)})
                    count += 1
    assert len(changed) == int(mapped)
    return {'files_compared': count, 'changed': changed}


try:
    parent = json.loads((PARENT / 'results.json').read_text())
    assert parent['status'] == 'isolated_preflight_failed_no_model_load'
    r['sources_before'] = sources(False)
    from fla_maca_device_mapping import apply_mapping
    r['mapping'] = apply_mapping(ENV, SITE / 'fla/utils.py', A / 'mapping_receipt.json')
    r['sources_after'] = sources(True)
    assert r['sources_after']['changed'][0]['actual_sha256'] == r['mapping']['after_sha256']
    probe = (PARENT / 'import_probe.py').read_text()
    probe += '''\nfrom fla import utils as u
assert u.device_platform == 'maca' and u.device_name == 'cuda'
assert not u.IS_NVIDIA and not hasattr(torch, 'maca')
assert u.device_torch_lib is torch.cuda
print(json.dumps({'real_backend':triton.runtime.driver.active.get_current_target().backend,
                  'torch_device':u.device_name,'IS_NVIDIA':u.IS_NVIDIA,'torch_maca_alias_created':hasattr(torch,'maca')}))
'''
    (A / 'import_probe.py').write_text(probe)
    start = time.time()
    with (A / 'import.log').open('w') as log:
        proc = subprocess.run([str(ENV / 'bin/python'), str(A / 'import_probe.py')], stdout=log,
            stderr=subprocess.STDOUT, timeout=90,
            env={**os.environ, 'MACA_PATH': '/opt/maca', 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    r['import'] = {'exit_code': proc.returncode, 'seconds': time.time() - start,
                   'log_sha256': sha((A / 'import.log').read_bytes())}
    r['status'] = 'device_mapping_import_passed_GPU_dispatch_not_validated' if proc.returncode == 0 else 'mapped_import_failed_no_model_load'
except Exception:
    r['status'] = 'mapping_probe_error'
    r['error'] = traceback.format_exc()
finally:
    (A / 'results.json').write_text(json.dumps(r, indent=2))
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in A.iterdir():
            if p.suffix in ['.py', '.json', '.log']:
                z.write(p, p.name)
    print(json.dumps({'status': r['status'], 'model_calls': 0}), flush=True)
