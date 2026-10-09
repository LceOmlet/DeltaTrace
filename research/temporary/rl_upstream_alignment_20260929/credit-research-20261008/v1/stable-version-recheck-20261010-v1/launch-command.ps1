@'
import sys,subprocess
sys.path.insert(0,'research/temporary/rl_upstream_alignment_20260929')
import stage_environment_entry as t
s="""source %s/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,hashlib,psutil,time,subprocess,os
from pathlib import Path
r=Path(%r);f=r/'runs/textcraft-formal-stable-20261009-v1'
p=psutil.Process(982372);assert abs(p.create_time()-1791553809.84)<.05
source=f/'source.json'; assert hashlib.sha256(source.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
m=json.loads(source.read_bytes())
script=r/'receipts/textcraft-formal-worker-runtime-20261009-v2/read_formal_worker_runtime_20261009.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='7705cae717a0ad3f3b4ecb8a4c23e39bca9c8dcd5c22490294b9bfb8a62f6f5a'
out=r/'receipts/stable-version-recheck-20261010-v1';out.mkdir(exist_ok=True)
assert not (out/'launch.json').exists(), 'reuse existing query instead of launching twice'
env=dict(os.environ);env.update(m['environment']);env['CUDA_VISIBLE_DEVICES']='-1'
args=[env['VENV_PYTHON'],str(script),str(out/'worker-runtime.json'),'127.0.0.1:58837','zagXiM']
with (out/'query.log').open('wb') as log:
 q=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=q.pid,birth=psutil.Process(q.pid).create_time(),unix=time.time(),formal_pid=p.pid,formal_birth=p.create_time(),script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),formal_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(out/'worker-runtime.json'),model_calls=0,optimizer_calls=0,runtime_patches=0)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY
"""%(t.ENTRY,t.ROOT)
r=subprocess.run(t.SSH+['bash','-s'],input=s.encode(),capture_output=True,timeout=35)
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));r.check_returncode()
'@ | C:/Users/Administrator/miniconda3/python.exe -X utf8 -
