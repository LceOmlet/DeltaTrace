source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-range-official-20261009-v1'
HASH='36657162a82b2ae9edd3ca13097d4e23d6c9d11710672adf3e3a20c31baf79bb'
COMMIT='26a6ba76b2f395b8680fccafa952eb71f8ee536a'
import json,os,subprocess,psutil,hashlib,re,time
from pathlib import Path
out=Path(OUT);assert hashlib.sha256((out/'check_fla_range_official.py').read_bytes()).hexdigest()==HASH
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'Diagnostic GPU occupied'
source_path=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
s=json.loads(source_path.read_bytes());env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/'check_fla_range_official.py'),'--operands',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt','--precast',ROOT+'/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt','--official',ROOT+'/receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py','--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as stream:process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),argv=argv,script_sha256=HASH,base_commit=COMMIT,devices=[4],launched_unix=time.time(),model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
