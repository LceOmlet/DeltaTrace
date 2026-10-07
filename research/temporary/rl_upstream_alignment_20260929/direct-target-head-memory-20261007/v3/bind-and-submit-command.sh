set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,hashlib,importlib.util,time
P=Path;root=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');candidate=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-head-memory-20261007-v2');base=candidate/'appworld';v=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-head-memory-20261007-v3/verification.json')
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
read=lambda p:json.loads(P(p).read_bytes())
r=read(v)
assert sha(v)=='eaeff74ea65f0783f10b8a77a4ed98d99215bb6876896cea17640ed4105599d4'
for item in r['records']:
 assert sha(item['path'])==item['sha256']
 x=read(item['path']);assert x['phase']=='head_memory_real_joint_complete' and x['within_owner_dtype_tolerance']
assert sha(r['diagnostic']['path'])==r['diagnostic']['sha256']
source=read(base/'source-template.json');prepared=read(base/'prepared.json')
assert source['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']==r['candidate_answer_sha256']
assert not (base/'prepared-before-head-verification.json').exists()
(base/'source-before-head-verification.json').write_bytes((base/'source-template.json').read_bytes())
(base/'prepared-before-head-verification.json').write_bytes((base/'prepared.json').read_bytes())
source['head_memory_verification']={'path':str(v),'sha256':sha(v),'scope':r['scope']}
source['source_bindings'][str(v)]=sha(v)
for item in r['records']:source['source_bindings'][item['path']]=item['sha256']
(base/'source-template.json').write_text(json.dumps(source,indent=2)+'\n')
prepared['source_template']['sha256']=sha(base/'source-template.json')
prepared['source_bindings']=source['source_bindings'];prepared['head_memory_verification']=source['head_memory_verification']
(base/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps({'prepared_sha256':sha(base/'prepared.json'),'source_sha256':sha(base/'source-template.json'),'code_commit':source['local_patch_commit'],'verification_sha256':sha(v),'observed_unix':time.time()}))

PY
"$VENV_PYTHON" /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-head-memory-20261007-v2/setup/submit_prepared_direct_targets.py --runtime-root /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922 --candidate-root /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-head-memory-20261007-v2 --tasks AppWorld --execute --receipt-dir /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-head-memory-20261007-v2-submit
