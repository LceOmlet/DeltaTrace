set -e
/opt/conda/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,runpy,os,time,psutil
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');d=root/'receipts/appworld-efficiency-20261007/restore-verified-vllm-31146-v1';d.mkdir(exist_ok=False)
sha=lambda x:hashlib.sha256(x).hexdigest()
active=json.loads((root/'active-training.json').read_text());j=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert j['pid']==532620 and j['status']=='stopped_installed_vllm_drift'
try:
 p=psutil.Process(j['pid']);assert p.create_time()!=j['observed_process_created_unix'] or p.status()==psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:pass
q={'started_unix':time.time(),'scope':'Restore exact previously verified bytes using original c9cd147 patch functions after known installed-file reversion; no package install, version upgrade, model/checkpoint load or new numerical implementation','previous_job':j,'files':[]}
for target,backup,patch,patchsha,before,after in [
('/opt/conda/lib/python3.12/site-packages/vllm_metax/device_allocator/cumem.py','metax-cumem-initialization.before.py','patch_metax_cumem_initialization.py','e6df3b1118b63b16dcde1af27a69efc526cb99340c5ff46a750bf496fb3fb771','c5581b1ea7e0fec3246b82a9317a5adc216e4b37a3cbe1825d87ffb910387365','7e557a8f5005112b37f64fef9ab95705749a45064943c251d51e11955a7bd528'),
('/opt/conda/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py','gpu-worker-before-weight-pool.py','patch_vllm_sleep.py','a3f3bb39d957405244955bbbe30d1fa67e0e0a02ee4c11eee638802a4573385c','9f9d0dbc36d4eae017959213e9f8fafaa023ab4b6a1a220c9dfd0de414e22319','37db3830453c3d8dc8852256c4763006e828b3e1da9f8f24d96c62135cc1d7c6')]:
 t=Path(target);b=root/'receipts/upstream-alignment-20260929'/backup;f=root/'releases/c9cd147/experiments/rl'/patch
 current=t.read_bytes();assert sha(current)==before and sha(b.read_bytes())==before
 assert sha(f.read_bytes())==patchsha
 restored=runpy.run_path(str(f))['patch_source'](b.read_text()).encode();assert sha(restored)==after
 compile(restored,str(t),'exec')
 (d/(t.parent.name+'-'+t.name+'.before')).write_bytes(current)
 tmp=t.with_name(t.name+'.verified-restore.tmp');tmp.write_bytes(restored);os.chmod(tmp,t.stat().st_mode);os.replace(tmp,t)
 assert sha(t.read_bytes())==after
 q['files'].append({'path':str(t),'backup':str(b),'patch':str(f),'patch_sha256':patchsha,'before_sha256':before,'after_sha256':after,'expected_verified_sha256':after})
q.update(completed_unix=time.time(),status='verified_bytes_restored_new_workers_required',numerical_receipt=str(root/'receipts/upstream-alignment-20260929/vllm-hybrid-owner.json'),numerical_receipt_sha256=sha((root/'receipts/upstream-alignment-20260929/vllm-hybrid-owner.json').read_bytes()))
(d/'restored.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
PY
