"""Launch the authorized two-change DTPO version through original VERL."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict_entry', HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=HERE, text=True).strip()
script = f'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=root/'receipts/textcraft-gradient-rootcause-20261013-v1'
prepared=out/'DT-group-multiplier-v40-prepared'
record=json.loads((prepared/'launch.json').read_bytes())
source=json.loads((prepared/'source.json').read_bytes())
verification=out/'DT-group-multiplier-interface-verification-v40.json'
checked=json.loads(verification.read_bytes())
assert checked['native_GRPO_valid_tokens_exact_broadcast'] and checked['DT_advantage_boundary_exact_passthrough']
assert checked['actual_zero_group_DT_path_zero'] and checked['raw_rewards_unchanged']
run=Path(record['run']);job=out/'DT-group-multiplier-v40-job.json'
assert not run.exists() and not job.exists(), 'Do not submit an existing job again'
for path,sha in record['source_bindings'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
(prepared/'before-launch-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]), 'GPU0/1 occupied'
protected=[]
for pid,birth,role in [(3174448,1791793629.73,'GRPO'),(2671393,1791949551.27,'PPO')]:
 p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
 protected.append(dict(pid=pid,birth=birth,role=role))
for pid,birth in [(3172194,1791793609.09),(2510247,1791907554.02)]:
 assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=birth
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
env.pop('RAY_ADDRESS',None);env.pop('RAY_TMPDIR',None)
assert 'VERL_GLOBAL_TOKEN_MEAN' not in env
record.update(local_patch_commit={commit!r},interface_verification=str(verification),
              interface_verification_SHA256=hashlib.sha256(verification.read_bytes()).hexdigest(),
              status='submitted_initialization_pending')
source.update(local_patch_commit={commit!r},deployment=dict(status=record['status']))
run.mkdir(parents=True)
(run/'effective-config.yaml').write_bytes((prepared/'effective-config.yaml').read_bytes())
with (run/'driver.stdout.log').open('wb') as stdout,(run/'driver.stderr.log').open('wb') as stderr:
 p=subprocess.Popen(record['argv'],cwd=env['VERL_ROOT'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
birth=psutil.Process(p.pid).create_time()
record.update(pid=p.pid,birth=birth,started_unix=time.time())
(run/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
identity=dict(pid=p.pid,birth=birth,run=str(run),physical_gpus=[0,1],version=record['version'],
              method='deltatrace',status=record['status'],started_unix=record['started_unix'],
              local_patch_commit={commit!r},upstream_commit='20bd331b',source_bindings=record['source_bindings'],
              launch_SHA256=hashlib.sha256((run/'launch.json').read_bytes()).hexdigest(),
              protected_jobs=protected,rootcause_repair_verified=False,hourly_automation='PAUSED')
job.write_text(json.dumps(identity,indent=2)+'\n')
(run/'job.json').write_bytes(job.read_bytes())
print(json.dumps(identity))
PY
'''
r = subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/'launch-DT-group-multiplier-v40.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-DT-group-multiplier-v40.stderr.txt').write_bytes(r.stderr)
if r.returncode:
    print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
result=json.loads(r.stdout)
(HERE/'DT-group-multiplier-v40-job.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('pid','birth','run','physical_gpus','version','status','local_patch_commit')}))
