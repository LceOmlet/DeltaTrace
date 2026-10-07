set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,subprocess,time,psutil,re
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');parent=root/'receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2';out=parent/'native-factual-jacobian'
assert not out.exists();out.mkdir()
helper=parent/'inspect_factual_jacobian.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()=='4c4fdbc01c4ff295b18a5097dd72ae7b512dfcad41ebfb6870ec8b389b46667c'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not any(re.match(r'^\|\s*4\s+\d+\s+\S',line) for line in physical.splitlines());(out/'before-physical.txt').write_text(physical)
assert psutil.Process(2833207).create_time()==1791370325.16
assert not any((root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
argv=[os.environ['VENV_PYTHON'],str(helper),'--source',str(root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'),'--launch',str(parent/'launch.json'),'--output',str(out/'results'),'--factual-jacobian']
with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4'),cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),argv=argv,code_commit='3c9c7081fd942654e0428f7ffb93b2c06eaae5f4',helper_sha256='4c4fdbc01c4ff295b18a5097dd72ae7b512dfcad41ebfb6870ec8b389b46667c',devices=[4],scope='Original native factual-endpoint VJP on saved actual single-deletion FLA operands; fixed incoming state and original output cotangent; operator-only forward/backward, no model/full DT/credit/profile/update/rollout/restore or tolerance change',formal_restart=False,text_update_released=False,production_profile_changed=False)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PY
