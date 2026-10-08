source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-seed-range-20261009-v2'
HASH='63eaffc9e5f03f27cc5af53e36518432cb1891745dc7deb4c8e0491701ded9af'
COMMIT='b47b713332053e6b12a7287222165244aba7fb60'
import json,os,subprocess,psutil,hashlib,re,time
from pathlib import Path
out=Path(OUT);assert hashlib.sha256((out/'check_fla_seed_range.py').read_bytes()).hexdigest()==HASH
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'Diagnostic GPU occupied'
source_path=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
s=json.loads(source_path.read_bytes());env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/'check_fla_seed_range.py'),'--source',str(source_path),'--operands',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt','--precast',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt','--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as stream:process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),argv=argv,script_sha256=HASH,base_commit=COMMIT,devices=[4],launched_unix=time.time(),model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
