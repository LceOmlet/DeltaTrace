"""Submit one bounded native endpoint diagnosis on currently free GPU4/5."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
remote=transport.ROOT+'/receipts/direct-target-reference-interaction-20261007-v1'
files=[HERE/'inspect_reference_context.py',AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
prepared=json.loads((HERE/'preparation.json').read_bytes())
hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
assert hashes['inspect_reference_context.py']==prepared['sha256']
assert hashes['inspect_extreme_endpoint.py']=='8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
subprocess.run(transport.SSH+['bash','-s'],input=('set -eu\nmkdir -p '+remote+'\n').encode(),check=True)
for path in files:subprocess.run(transport.SCP+[str(path),transport.SSH[-1]+':'+remote+'/'+path.name],check=True)
code=r'''
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path(@ROOT@);out=Path(@OUT@)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(source_path.read_bytes())
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
for name,sha in @HASHES@.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
assert not (out/'launch.json').exists(),'Do not launch this diagnosis twice'
assert psutil.Process(2833207).create_time()==1791370325.16
for rank in (0,1):
 a=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
 assert not (a/('rank'+str(rank)+'-release-update')).exists()
for pid in (2786671,3714027):assert not psutil.pid_exists(pid)
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Selected devices are occupied'
(out/'before-physical.txt').write_text(physical)
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_reference_context.py'),'--source',str(source_path),'--output',str(out/'results')]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=argv,scripts=@HASHES@,source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),DT_calls=0,optimizer_steps=0,formal_restart=False,update_released=False)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('@ROOT@',repr(transport.ROOT)).replace('@OUT@',repr(remote)).replace('@HASHES@',repr(hashes))
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
(HERE/'launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
(HERE/'launch.stderr.txt').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode();record=json.loads(result.stdout)
(HERE/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
