set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

import hashlib,json,os,pathlib,re,subprocess,time
import psutil
P=pathlib.Path; root=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'); out=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-memory-capacity-20261008-v1')
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
preparation=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
for key,expected in [('changed_file','changed_sha256'),('candidate_environment','candidate_environment_sha256')]:
 assert hashlib.sha256(P(preparation[key]).read_bytes()).hexdigest()==preparation[expected]
hashes={'check_capacity_lifetime.py': '060070d993cef62fae79150b7e2e6ffc4d0521b7fd13e383181d285eac291d3a', 'profile_existing_offload.py': '732f434a1e3c6deea46e6c5c9fab68b481d15be59e477d4f03147e13ec2892c7', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}
for name,expected in hashes.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK=source['startup_options']['env.env_name'],DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=preparation['candidate_dt_root'],DT_ENVIRONMENT_JSON=preparation['candidate_environment'])
dt=P(env['DT_ROOT']); qwen=json.loads(P(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
failed=root/'receipts/direct-target-native-mlp-memory-20261007-v3/failed-inputs'
if not True:
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
 record=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),unix=time.time(),argv=argv,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),scripts=hashes,diagnostic_commit='6cfa4a9d7a48ac420016e5c02a7b71c3bf72b182',candidate=preparation,devices=[4,5],checkpoint_restore=False,optimizer_steps=0,formal_restart=False,text_update_released=False)
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))

PY
