"""Stage/launch one bounded existing-rule diagnosis; no production deployment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
OUT = transport.ROOT + '/receipts/direct-target-existing-pv-rule-20261008-v1'
files = [HERE / 'compare_existing_pv_rule.py',
         AUDIT / 'direct-target-native-mlp-memory-20261007/v2/profile_existing_offload.py',
         AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
args = parser.parse_args()
for p in files:
    compile(p.read_bytes(), str(p), 'exec')
subprocess.run(transport.SSH + ['bash', '-s'], input=('set -eu\nmkdir -p ' + OUT + '\n').encode(), check=True)
for p in files:
    subprocess.run(transport.SCP + [str(p), transport.SSH[-1] + ':' + OUT + '/' + p.name], check=True)
code = r"""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
import psutil
root=Path(@ROOT@);out=Path(@OUT@);hashes=@HASHES@
for name,expected in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_sha=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
for key,expected in [('changed_file','changed_sha256'),('candidate_environment','candidate_environment_sha256')]:
 assert hashlib.sha256(Path(candidate[key]).read_bytes()).hexdigest()==candidate[expected]
native=root/'receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank1-readout-native-batch-16.pt'
assert hashlib.sha256(native.read_bytes()).hexdigest()=='3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a'
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK=source['startup_options']['env.env_name'],DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=candidate['candidate_dt_root'],DT_ENVIRONMENT_JSON=candidate['candidate_environment'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
if not @LAUNCH@:
 env['CUDA_VISIBLE_DEVICES']=''
 script='''import json,torch
from pathlib import Path
from reward_readout import DirectActionTargetReadout
d=torch.load(NATIVE,map_location='cpu',weights_only=False);result=[]
for item in sorted(d['rows'],key=lambda x:x['batch_row']):
 p=DirectActionTargetReadout._prepare_row(item['row'],0)
 assert torch.equal(p['selected'],item['selected']) and torch.equal(p['case']['target_ids'],item['case']['target_ids']) and p['target_offsets']==item['target_offsets']
 result.append(dict(uid=item['traj_uid'],length=p['selected'].numel(),targets=len(p['target_offsets'])))
print(json.dumps(dict(rows=result,cuda_initialized=torch.cuda.is_initialized(),scope='CPU input identity only, no model/DT/update')))
'''.replace('NATIVE',repr(str(native)))
 r=subprocess.run([env['VENV_PYTHON'],'-c',script],env=env,cwd=out,capture_output=True)
 (out/'prepare.stdout.txt').write_bytes(r.stdout);(out/'prepare.stderr.txt').write_bytes(r.stderr);r.check_returncode()
 (out/'prepared.json').write_bytes(r.stdout);print(r.stdout.decode())
else:
 assert (out/'prepared.json').exists() and not (out/'launch.json').exists()
 assert psutil.Process(2833207).create_time()==1791370325.16
 held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
 assert not any((held/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
 physical=subprocess.run(['mx-smi'],capture_output=True,check=True).stdout;(out/'before-physical.txt').write_bytes(physical)
 assert not any(re.match(r'^\|\s*[45]\s+\d+\s+\S',line) for line in physical.decode(errors='replace').splitlines())
 env['CUDA_VISIBLE_DEVICES']='4,5'
 argv=[env['VENV_PYTHON'],str(out/'compare_existing_pv_rule.py'),'--source',str(source_path),'--native',str(native),'--output',str(out/'results')]
 with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),code_commit=@COMMIT@,argv=argv,scripts=hashes,source_sha256=source_sha,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),memory_candidate=candidate,devices=[4,5],DT_calls_per_rank=2,scope='Existing content1/content0 diagnostic only, no profile deployment, no parameter/optimizer/scheduler updates, no rollout or checkpoint restore',formal_restart=False,credit_repaired=False,text_update_released=False)
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
"""
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
for k, v in dict(ROOT=transport.ROOT, OUT=OUT, HASHES={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, LAUNCH=args.launch, COMMIT=commit).items():
    code = code.replace('@' + k + '@', repr(v))
command = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
phase = 'launch' if args.launch else 'prepare'
(HERE / (phase + '-command.sh')).write_text(command, encoding='utf8', newline='\n')
r = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(), capture_output=True)
(HERE / (phase + '.stdout.txt')).write_bytes(r.stdout)
(HERE / (phase + '.stderr.txt')).write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
if args.launch:
    (HERE / 'launch.json').write_bytes(r.stdout)
else:
    (HERE / 'prepared.json').write_bytes(r.stdout)
