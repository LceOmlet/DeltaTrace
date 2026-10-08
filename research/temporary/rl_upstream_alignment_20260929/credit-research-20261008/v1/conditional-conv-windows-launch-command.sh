source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-conv-windows-20261009-v1'
FILES={'conditional_conv_windows.py': 'cb336dc1489ff45cfb8fa34334b90e04fe5848f187939697f42a7ed5ad33cbfc', 'check_conditional_conv_windows.py': 'f820fbc3099a77a9db98ab3212634feefa88da495004d94bedec34077409cfa4', 'original_test_causal_conv1d_v150.py': 'c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf'}
COMMIT='1d908254aaaab033a78d45d01afeb1f2e6ca12e5'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
for name,digest in FILES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
assert not (out/'launch.json').exists(), 'Inspect the original job; do not duplicate it'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
device=next(i for i in (4,5) if not re.search(r'^\|\s*'+str(i)+r'\s+\d+\s+\S',physical,re.M))
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']=str(device)
env['PYTHONPATH']=':'.join([str(out),s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_conditional_conv_windows.py'),
 '--operands',ROOT+'/receipts/gdn-cached-conv-interface-20261007/base-native-b8-long-v1/rank0.pt',
 '--official',str(out/'original_test_causal_conv1d_v150.py'),'--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=FILES,
 base_commit=COMMIT,source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 devices=[device],launched_unix=time.time(),wall_bound_seconds=240,
 model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
