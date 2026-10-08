"""Stage the bounded, original-owner query diagnostic; never launch training."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, ROOT, SCP, SSH


def main():
    names = ('check_native_conditional_queries.py', 'native_conditional_queries.py')
    files = {}
    for name in names:
        data = (HERE/name).read_bytes()
        ast.parse(data)
        files[name] = hashlib.sha256(data).hexdigest()
    remote = ROOT+'/receipts/native-conditional-queries-20261009-v1'
    subprocess.run(SSH+['mkdir', '-p', remote], check=True, timeout=30)
    subprocess.run(SCP+[str(HERE/name) for name in names]+[SSH[-1]+':'+remote+'/'], check=True, timeout=40)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    body = '''import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
for name,digest in FILES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
assert not (out/'launch.json').exists(), 'Do not duplicate a diagnostic'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
assert not re.search(r'^\\|\\s*4\\s+\\d+\\s+\\S',physical,re.M), 'Diagnostic GPU occupied'
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([str(out),s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_native_conditional_queries.py'),
 '--operands',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt',
 '--precast',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt',
 '--official',ROOT+'/receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py',
 '--output',str(out/'result.json')]
for key in ('--operands','--precast','--official'):
 assert Path(argv[argv.index(key)+1]).is_file(), key
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=FILES,
 base_commit=COMMIT,source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 devices=[4],launched_unix=time.time(),wall_bound_seconds=240,
 model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,
 physical_before=physical,scope='Original o/dq/dk/dv/db checks, saved real B4 x 199 x head8')
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''
    script = f'source {ENTRY}/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
    script += 'ROOT='+repr(ROOT)+'\nOUT='+repr(remote)+'\nFILES='+repr(files)+'\nCOMMIT='+repr(commit)+'\n'+body+'\nPY\n'
    (HERE/'native-conditional-queries-launch-command.sh').write_bytes(script.encode())
    r = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=45)
    (HERE/'native-conditional-queries-launch.stderr.txt').write_bytes(r.stderr)
    r.check_returncode()
    result = json.loads(r.stdout)
    (HERE/'native-conditional-queries-launch.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'physical_before'}))


if __name__ == '__main__':
    main()
