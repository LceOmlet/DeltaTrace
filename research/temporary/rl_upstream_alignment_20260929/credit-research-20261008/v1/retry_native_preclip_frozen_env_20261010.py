"""Repair the failed diagnostic sender by reusing the formal owner's frozen env.

The first sender died in client argument validation before any worker RPC.
Keep that failure; do not restart any training process or change the observer.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
query=r'''
import json,os,psutil,time
from pathlib import Path
import ray
import capture_native_preclip_20261010 as observer
ray.cloudpickle.register_pickle_by_value(observer)
out=Path(__file__).parent
ray.init(address='127.0.0.1:58837',log_to_driver=False,ignore_reinit_error=False)
named=[x for x in ray.util.list_named_actors(all_namespaces=True)
 if x['name'].startswith('zagXiMWorkerDict') and 'register_center' not in x['name']]
assert len(named)==2,named
record=dict(pid=os.getpid(),birth=psutil.Process().create_time(),started_unix=time.time(),named_actors=named,complete=False)
path=out/'query-result-v2.json';path.write_text(json.dumps(record,indent=2)+'\n')
try:
 refs=[ray.get_actor(x['name'],namespace=x['namespace']).execute_with_func_generator.remote(observer.install) for x in named]
 record.update(results=ray.get(refs),complete=True,completed_unix=time.time())
 path.write_text(json.dumps(record,indent=2)+'\n')
except Exception as error:
 record.update(error=repr(error),failed_unix=time.time())
 path.write_text(json.dumps(record,indent=2)+'\n')
 raise
finally:ray.shutdown()
'''
code=r'''
import hashlib,json,os,psutil,subprocess,sys,time
from pathlib import Path
root=Path(ROOT);out=root/'receipts/textcraft-native-actor-incidents-20261010-v1/preclip-v1'
driver=psutil.Process(982372);assert driver.create_time()==1791553809.84
try:
 old=psutil.Process(1706148)
 assert old.create_time()!=1791601350.47 or old.status()==psutil.STATUS_ZOMBIE,'Original sender still live'
except psutil.NoSuchProcess:pass
assert 'TypeError: too many positional arguments' in (out/'query.log').read_text()
assert not list(out.parent.glob('rank*-pid*/preclip-installation.json'))
source_path=root/'runs/textcraft-formal-stable-20261009-v1/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
source=json.loads(source_path.read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=source['pythonpath']
script=out/'install_preclip_query_v2.py'
assert not script.exists(),'Inspect the existing v2 handle; do not resubmit'
script.write_text(QUERY)
with (out/'query-v2.log').open('xb') as stream:
 p=subprocess.Popen([env['VENV_PYTHON'],str(script)],env=env,cwd=out,
     stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),path=str(script),
 script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),out=str(out),
 frozen_source_path=str(source_path),frozen_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
 python=env['VENV_PYTHON'],pythonpath=env['PYTHONPATH'],status='submitted_frozen_environment_not_verified',
 first_sender=dict(pid=1706148,birth=1791601350.47,terminal=True,
   failure='Ray client signature validation; frozen PYTHONPATH was not inherited; no worker RPC dispatched'))
(out/'launch-v2.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('ROOT',repr(transport.ROOT),1).replace('QUERY',repr(query),1)
command='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=45)
r.check_returncode();d=json.loads(r.stdout)
out=HERE/'direct-credit-records-20261009-v1/native-preclip-stage-frozen-env-v2-20261010.json'
out.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(out),**d)))
