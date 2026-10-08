source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-memory-finite-20261009-v1'
FILES={'check_conditional_memory_finite.py': 'ddc6aa3bcd7f22b7271b5eaa1c970b34665b88f62a768cf7005006331185f908', 'native_conditional_queries.py': '9f8e8c59ee372cdb5e58e835c0de351fa3c9f7829cbdcb3af45c9057e9254aa5', 'conditional_window_memory.py': '8e4c0502c2f89ce2074b5ed0a92dbd0725f2ba4ea9dd86d822f4b0807d73e4d1'}
COMMIT='30bb2ded85b8e073bcbbef7ff85ffadb6c8ce091'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
for name,digest in FILES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
assert not (out/'launch.json').exists(), 'Do not duplicate a diagnostic'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M), 'Diagnostic GPU occupied'
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([str(out),s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_conditional_memory_finite.py'),
 '--original',ROOT+'/receipts/native-conditional-queries-20261009-v1',
 '--output',str(out/'result.json')]
assert Path(argv[argv.index('--original')+1]+'/result.json').is_file()
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=FILES,
 base_commit=COMMIT,source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 devices=[4],launched_unix=time.time(),wall_bound_seconds=240,
 model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,
 physical_before=physical,scope='Nonzero local memory identity on all199 sources, original native/FP32 FLA')
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
