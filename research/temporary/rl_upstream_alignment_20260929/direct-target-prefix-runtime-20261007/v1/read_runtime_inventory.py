"""Read-only current resources and frozen owner inputs; never launches work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_transport', AUDIT / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

CODE = r'''
from pathlib import Path
import hashlib,json,os,psutil,shutil,subprocess,time
root=Path(@ROOT@)
def identity(path):
 p=Path(path);result=dict(path=str(p),exists=p.exists())
 if p.is_file():
  value=p.read_bytes();result.update(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value))
 return result
record=dict(observed_unix=time.time(),read_only=True,host_memory=dict(psutil.virtual_memory()._asdict()),disk=dict(shutil.disk_usage(root)._asdict()))
record['cgroup']={str(p):p.read_text() for p in [Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.stat')] if p.exists()}
smi=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15)
record['physical_mx_smi']=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr)
record['prior_processes']=[]
for pid,birth in [(110053,1791344324.6),(2001805,1791362313.39)]:
 try:
  process=psutil.Process(pid);row=dict(pid=pid,expected_birth=birth,actual_birth=process.create_time(),status=process.status(),name=process.name())
 except psutil.NoSuchProcess:row=dict(pid=pid,expected_birth=birth,status='NoSuchProcess')
 record['prior_processes'].append(row)
relative=[
 'active-training.json','active-source.json',
 'receipts/environment-only-20260930/entry/metax-entry.env.sh',
 'receipts/direct-target-numerics-20261007-v4/actual-direct-target-inputs.json',
 'receipts/direct-target-numerics-20261007-v4/run_direct_target_numeric_owner.py',
 'receipts/direct-target-mlp-token-chunk-20261007-v1/actual-direct-target-inputs.json',
 'receipts/direct-target-mlp-token-chunk-20261007-v1/run_direct_target_numeric_owner.py',
 'candidates/direct-target-causal-prefix-20261007-v2/appworld/entry/reward_readout.py',
 'candidates/direct-target-causal-prefix-20261007-v2/appworld/entry/deltatrace_rollout.py',
 'candidates/direct-target-causal-prefix-20261007-v2/appworld/entry/native_prefix_leases.py',
 'candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py',
 'candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_decoder_finite.py',
 'runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt/source.json',
 'runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt/rollouts/0.jsonl',
]
record['files']=[identity(root/p) for p in relative]
print(json.dumps(record),flush=True)
'''

if __name__ == '__main__':
    code = CODE.replace('@ROOT@', repr(stage.ROOT))
    shell = 'set -eu\nsource ' + stage.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
    started = time.time()
    result = subprocess.run(stage.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True, timeout=60)
    (HERE / 'inventory.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    record = json.loads(next(line for line in reversed(result.stdout.decode().splitlines()) if line.startswith('{')))
    path = HERE / ('inventory-' + str(int(record['observed_unix'])) + '.json')
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=time.time()-started, physical_mx_smi=record['physical_mx_smi'], prior_processes=record['prior_processes'], files=record['files'])))
