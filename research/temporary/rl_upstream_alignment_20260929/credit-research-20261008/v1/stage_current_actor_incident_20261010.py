"""Stage a passive observer for the existing native TextCraft workers only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
source=HERE/'capture_current_actor_incident_20261010.py'
out=transport.ROOT+'/receipts/textcraft-native-actor-incidents-20261010-v1'
expected=hashlib.sha256(source.read_bytes()).hexdigest()
subprocess.run(transport.SSH+['mkdir','-p',out],check=True,timeout=30)
subprocess.run(transport.SCP+[str(source),transport.SSH[-1]+':'+out+'/'+source.name],check=True,timeout=40)
code=r'''
import ast,hashlib,inspect,json,os,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT);expected_sha=EXPECTED;source=out/'capture_current_actor_incident_20261010.py'
assert psutil.Process(982372).create_time()==1791553809.84
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha
ast.parse(source.read_bytes())
launch=out/'launch.json'
if launch.exists():
 print(launch.read_text());raise SystemExit(0)
existing=out/'installation-results.json'
if existing.exists():
 result=json.loads(existing.read_bytes())
 pid=result['pid'];birth=result['birth']
 try:alive=psutil.Process(pid).create_time()==birth
 except psutil.NoSuchProcess:alive=False
 record=dict(unix=time.time(),query_pid=pid,query_birth=birth,query_alive=alive,
  formal_pid=982372,formal_birth=1791553809.84,source_path=str(source),source_sha256=expected_sha,
  recovered_existing_submission=True,result=str(existing),log=str(out/'query.log'),
  installation_complete=result['complete'],status='recovered_existing_query_without_resubmission',
  controller_error='Initial launch metadata NameError after Popen; original observer process was already started.',
  loss_changes=0,configuration_changes=0,new_model_calls=0,training_limits_added=0)
 launch.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record));raise SystemExit(0)
formal=root/'runs/textcraft-formal-stable-20261009-v1'
assert hashlib.sha256((formal/'source.json').read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads((formal/'source.json').read_bytes())
env=dict(os.environ,**s['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=s['pythonpath']
check='import inspect,torch; from torch.optim import Optimizer; print(inspect.signature(Optimizer.register_step_post_hook)); print(torch.cuda.is_initialized())'
probe=subprocess.run([env['VENV_PYTHON'],'-c',check],env=env,capture_output=True,text=True,timeout=30)
assert probe.returncode==0,probe.stderr
assert probe.stdout.strip().endswith('False'),probe.stdout
result=out/'installation-results.json'
argv=[env['VENV_PYTHON'],str(source),str(result),'127.0.0.1:58837','zagXiMWorkerDict']
log=out/'query.log'
with log.open('xb') as stream:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(unix=time.time(),query_pid=p.pid,query_birth=psutil.Process(p.pid).create_time(),
 formal_pid=982372,formal_birth=1791553809.84,source_path=str(source),source_sha256=expected_sha,
 native_source_sha256='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
 upstream_commit=s['upstream_commit'],argv=argv,log=str(log),result=str(result),
 CPU_owner_interface_check=probe.stdout,CPU_check_CUDA_initialized=False,
 status='submitted_to_original_worker_queue_not_yet_installed',loss_changes=0,
 configuration_changes=0,new_model_calls=0,training_limits_added=0,checkpoint_restore=False)
launch.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('ROOT',repr(transport.ROOT),1).replace('OUT',repr(out),1).replace('EXPECTED',repr(expected),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=50)
(HERE/'direct-credit-records-20261009-v1/native-actor-observer-stage.stderr').write_bytes(r.stderr)
r.check_returncode()
d=json.loads(r.stdout)
path=HERE/'direct-credit-records-20261009-v1/native-actor-observer-stage.json'
path.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(path),**d)))
