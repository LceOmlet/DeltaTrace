source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/native-context-readouts-20261009-v3'
HASH='8dc07ffbf54a4bb7aa396cd6c78f567a0fee4cbf78753ccd407714b99c734773'
COMMIT='30bb2ded85b8e073bcbbef7ff85ffadb6c8ce091'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
assert hashlib.sha256((out/'check_native_context_readouts.py').read_bytes()).hexdigest()==HASH
assert not (out/'launch.json').exists(), 'Do not duplicate an existing diagnostic'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M), 'Diagnostic GPU occupied'
source_path=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source_path.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_native_context_readouts.py'),
 '--operands',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-first-nonfinite.pt',
 '--precast',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt',
 '--official',ROOT+'/receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py',
 '--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=HASH,
 base_commit=COMMIT,devices=[4],launched_unix=time.time(),model_calls=0,DT_calls=0,optimizer=0,
 formal_restart=False,production_modified=False,physical_before=physical,
 wall_bound_seconds=240,expected_scope='Saved B4 x 8 heads x 128 tokens, native output/dq/dv only')
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
