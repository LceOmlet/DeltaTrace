"""Launch the bounded original-assertion operand test on currently idle GPU4."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_transport',HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote = transport.ROOT+'/receipts/current-extreme-official-block-dtype-20261008-v1'
helper = HERE/'check_current_block_official_dtypes.py'
digest = hashlib.sha256(helper.read_bytes()).hexdigest()
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
subprocess.run(transport.SSH+['mkdir','-p',remote],check=True)
subprocess.run(transport.SCP+[str(helper),transport.SSH[-1]+':'+remote+'/'+helper.name],check=True)
script = fr'''set -eu
source {transport.ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PYCODE'
import hashlib,json,os,re,subprocess,time,psutil
from pathlib import Path
out=Path('{remote}');helper=out/'{helper.name}'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='{digest}'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not any(re.match(r'^\|\s*4\s+\d+\s+\S',line) for line in physical.splitlines())
(out/'before-physical.txt').write_text(physical)
assert psutil.Process(2833207).create_time()==1791370325.16
hold=Path('{transport.ROOT}/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt')
assert not any((hold/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
argv=[os.environ['VENV_PYTHON'],str(helper)]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4'),cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),argv=argv,
 code_commit='{commit}',helper_sha256='{digest}',devices=[4],scope='Actual current GDN30 first changed block, all32 heads; original saved dtype harness/reference/assertions, no model/full DT/update/rollout/restore',formal_restart=False,text_update_released=False)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PYCODE
'''
(HERE/'block-dtype-launch-command.sh').write_text(script,encoding='utf8')
run = subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True)
(HERE/'block-dtype-launch.stdout.txt').write_bytes(run.stdout)
(HERE/'block-dtype-launch.stderr.txt').write_bytes(run.stderr)
run.check_returncode()
launch = json.loads(run.stdout)
(HERE/'block-dtype-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
