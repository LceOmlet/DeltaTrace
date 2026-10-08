"""Launch one bounded, original-operator diagnostic on saved real operands."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ENTRY, ROOT, SCP, SSH


def main():
    worker = HERE/'check_native_context_readouts.py'
    ast.parse(worker.read_bytes())
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = ROOT+'/receipts/native-context-readouts-20261009-v1'
    subprocess.run(SSH+['mkdir', '-p', remote], check=True, timeout=30)
    subprocess.run(SCP+[str(worker), SSH[-1]+':'+remote+'/'+worker.name], check=True, timeout=40)
    body = '''import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
assert hashlib.sha256((out/'check_native_context_readouts.py').read_bytes()).hexdigest()==HASH
assert not (out/'launch.json').exists(), 'Do not duplicate an existing diagnostic'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
assert not re.search(r'^\\|\\s*4\\s+\\d+\\s+\\S',physical,re.M), 'Diagnostic GPU occupied'
source_path=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source_path.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_native_context_readouts.py'),
 '--operands',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt',
 '--precast',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt',
 '--official',ROOT+'/receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py',
 '--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=HASH,
 base_commit=COMMIT,devices=[4],launched_unix=time.time(),model_calls=0,DT_calls=0,optimizer=0,
 formal_restart=False,production_modified=False,physical_before=physical,
 wall_bound_seconds=240,expected_scope='Saved B4 x 8 heads x 128 tokens, native output/dq/dv only')
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''
    script = f'source {ENTRY}/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
    script += 'ROOT='+repr(ROOT)+'\nOUT='+repr(remote)+'\nHASH='+repr(digest)+'\nCOMMIT='+repr(commit)+'\n'+body+'\nPY\n'
    (HERE/'native-context-readouts-launch-command.sh').write_bytes(script.encode())
    result = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=45)
    (HERE/'native-context-readouts-launch.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    launch = json.loads(result.stdout)
    (HERE/'native-context-readouts-launch.json').write_text(json.dumps(launch, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: v for k, v in launch.items() if k != 'physical_before'}))


if __name__ == '__main__':
    main()
