"""Run the capacity/lifetime diagnosis once, on the existing isolated cards."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
REMOTE = transport.ROOT+'/receipts/direct-target-memory-capacity-20261008-v1'
files = [HERE/'check_capacity_lifetime.py',
    AUDIT/'direct-target-native-mlp-memory-20261007/v2/profile_existing_offload.py',
    AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
args = parser.parse_args()
for path in files:
    compile(path.read_bytes(), str(path), 'exec')
r = subprocess.run(transport.SSH+['bash','-s'], input=('set -eu\nmkdir -p '+REMOTE+'\n').encode(), capture_output=True)
r.check_returncode()
for path in files:
    subprocess.run(transport.SCP+[str(path), transport.SSH[-1]+':'+REMOTE+'/'+path.name], capture_output=True, check=True)
code = r"""
import hashlib,json,os,pathlib,re,subprocess,time
import psutil
P=pathlib.Path; root=P(@ROOT@); out=P(@OUT@)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
preparation=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
for key,expected in [('changed_file','changed_sha256'),('candidate_environment','candidate_environment_sha256')]:
 assert hashlib.sha256(P(preparation[key]).read_bytes()).hexdigest()==preparation[expected]
hashes=@HASHES@
for name,expected in hashes.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK=source['startup_options']['env.env_name'],DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=preparation['candidate_dt_root'],DT_ENVIRONMENT_JSON=preparation['candidate_environment'])
dt=P(env['DT_ROOT']); qwen=json.loads(P(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
failed=root/'receipts/direct-target-native-mlp-memory-20261007-v3/failed-inputs'
if not @LAUNCH@:
 env['CUDA_VISIBLE_DEVICES']=''
 script='''import json,torch
from pathlib import Path
from check_capacity_lifetime import capacity_row
from reward_readout import DirectActionTargetReadout
result=[]
for rank in (0,1):
 saved=torch.load(Path(FAILED)/('rank'+str(rank)+'-failed-batch.pt'),map_location='cpu',weights_only=False)
 values=[capacity_row(row,DirectActionTargetReadout._prepare_row) for row in saved['rows']]
 result.append(dict(rank=rank,rows=[v[1] for v in values]))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(ranks=result,cuda_initialized=False)))
'''.replace('FAILED',repr(str(failed)))
 result=subprocess.run([env['VENV_PYTHON'],'-c',script],env=env,cwd=out,capture_output=True)
 (out/'prepare.stdout.txt').write_bytes(result.stdout);(out/'prepare.stderr.txt').write_bytes(result.stderr)
 result.check_returncode()
 (out/'prepared.json').write_bytes(result.stdout)
 print(result.stdout.decode())
else:
 assert (out/'prepared.json').exists()
 assert not (out/'launch.json').exists(),'Never launch twice'
 assert psutil.Process(2833207).create_time()==1791370325.16
 held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
 for rank in (0,1):
  assert not (held/('rank'+str(rank)+'-release-update')).exists()
 assert all(not psutil.pid_exists(pid) for pid in (2786671,2792547,2794018))
 physical=subprocess.run(['mx-smi'],capture_output=True,check=True).stdout
 (out/'before-physical.txt').write_bytes(physical)
 for line in physical.decode(errors='replace').splitlines():
  assert not re.match(r'^\|\s*[45]\s+\d+\s+\S',line),line
 env['CUDA_VISIBLE_DEVICES']='4,5'
 argv=[env['VENV_PYTHON'],str(out/'check_capacity_lifetime.py'),'--source',str(source_path),
       '--native',str(root/'receipts/direct-target-native-mlp-memory-20261007-v6/results/rank1-failed-actual-complete.pt'),
       '--output',str(out/'results'),'--failed-batch',str(failed),'--with-vllm']
 with (out/'driver.log').open('xb') as stream:
  process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),unix=time.time(),argv=argv,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),scripts=hashes,diagnostic_commit=@COMMIT@,candidate=preparation,devices=[4,5],checkpoint_restore=False,optimizer_steps=0,formal_restart=False,text_update_released=False)
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))
"""
commit = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
for name, value in {'ROOT':transport.ROOT,'OUT':REMOTE,'HASHES':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files},'LAUNCH':args.launch,'COMMIT':commit}.items():
    code = code.replace('@'+name+'@',repr(value))
command = 'set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
phase = 'launch' if args.launch else 'prepare'
(HERE/(phase+'-command.sh')).write_text(command,encoding='utf8',newline='\n')
result = subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
(HERE/(phase+'.stdout.txt')).write_bytes(result.stdout)
(HERE/(phase+'.stderr.txt')).write_bytes(result.stderr)
print(result.stdout.decode(errors='replace'))
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
