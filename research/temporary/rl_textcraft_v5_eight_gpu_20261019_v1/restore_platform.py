"""Restore only three demonstrated reset files to their accepted exact hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from prepare_campaign import ROOT,AUDIT,sha

physical=subprocess.check_output(['mx-smi'],text=True)
assert 'no process found' in physical, 'Do not change the runtime under occupied GPUs'
(AUDIT/'physical-before-platform-restore.txt').write_text(physical)
records=[]
for module,target,before,after in [
    ('patch_metax_cumem_initialization','/opt/conda/lib/python3.12/site-packages/vllm_metax/device_allocator/cumem.py',
     'c5581b1ea7e0fec3246b82a9317a5adc216e4b37a3cbe1825d87ffb910387365',
     '7e557a8f5005112b37f64fef9ab95705749a45064943c251d51e11955a7bd528'),
    ('patch_vllm_sleep','/opt/conda/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py',
     '9f9d0dbc36d4eae017959213e9f8fafaa023ab4b6a1a220c9dfd0de414e22319',
     '37db3830453c3d8dc8852256c4763006e828b3e1da9f8f24d96c62135cc1d7c6')]:
    owner=ROOT/'releases/c9cd147/experiments/rl'/(module+'.py')
    p=Path(target);actual=sha(p)
    assert actual in (before,after),(target,actual)
    if actual!=after:
        shutil.copyfile(p,AUDIT/(module+'.before.py'))
        spec=importlib.util.spec_from_file_location(module,owner)
        patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
        changed=patch.patch_source(p.read_text())
        assert hashlib.sha256(changed.encode()).hexdigest()==after
        p.write_text(changed)
    records.append(dict(path=target,before=actual,after=sha(p),owner=str(owner),owner_SHA256=sha(owner)))
receipt=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/native-resource-owner-deployed.json'
item=json.loads(receipt.read_bytes());target=Path(item['owner_path']);actual=sha(target)
assert actual in (item['original_SHA256'],item['candidate_SHA256'])
assert sha(item['candidate'])==item['candidate_SHA256']
if actual!=item['candidate_SHA256']:
    shutil.copyfile(target,AUDIT/'gpu_model_runner.before.py')
    shutil.copyfile(item['candidate'],target)
records.append(dict(path=str(target),before=actual,after=sha(target),accepted_deployment_receipt=str(receipt)))
(AUDIT/'restored-platform-files.json').write_text(json.dumps(dict(status='accepted_exact_sources_restored',
    changes=records,packages_installed=0,caches_cleared=0,model_calls=0),indent=2)+'\n')
print(json.dumps(dict(status='accepted_exact_sources_restored',files=len(records),packages_installed=0,caches_cleared=0)))
