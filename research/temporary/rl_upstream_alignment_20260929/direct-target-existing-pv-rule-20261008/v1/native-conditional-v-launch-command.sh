set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,subprocess,time,psutil,re
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');parent=root/'receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2';out=parent/'native-conditional-v'
assert not out.exists();out.mkdir()
helper=parent/'inspect_conditional_v.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()=='6b3a5564c68f24b8342a720fe738043e352a7a7f57507ddc44c16c8ccb10ac29'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not any(re.match(r'^\|\s*4\s+\d+\s+\S',line) for line in physical.splitlines());(out/'before-physical.txt').write_text(physical)
assert psutil.Process(2833207).create_time()==1791370325.16
assert not any((root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
argv=[os.environ['VENV_PYTHON'],str(helper),'--source',str(root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'),'--launch',str(parent/'launch.json'),'--output',str(out/'results'),'--conditional-v-only']
with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4'),cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),argv=argv,code_commit='9738653b4a01bf014a623945f58260999a983a5b',helper_sha256='6b3a5564c68f24b8342a720fe738043e352a7a7f57507ddc44c16c8ccb10ac29',devices=[4],scope='Conditional V only on saved actual single V operands, other inputs factual; unchanged native FLA and existing forward coefficients; no model/full-DT/rollout/actor/training-backward/optimizer/restore or production rule change',formal_restart=False,text_update_released=False,production_profile_changed=False)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PY
