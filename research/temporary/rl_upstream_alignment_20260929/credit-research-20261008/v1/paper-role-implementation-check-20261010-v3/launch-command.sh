source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
REMOTE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-role-implementation-check-20261010-v3'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
HASHES={'probe_paper_role_dt_20261010_v3.py': '31165846edd069e496b2f0b3d0edd11d6a90d24ccf88c0cd929cba7160bb8703', 'paper-inputs.json': '2f17897f9f87c662432b86e6c46a76fb113a1e483a21a2e1e223dd6961badd01'}

import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(REMOTE);root=Path(ROOT)
assert not (out/'launch.json').exists(),'Do not duplicate this diagnostic'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 is occupied'
assert psutil.Process(982372).create_time()==1791553809.84
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,value in HASHES.items():assert sha(out/name)==value
ast.parse((out/'probe_paper_role_dt_20261010_v3.py').read_text())
p=root/'runs/textcraft-formal-stable-20261009-v1/source.json'
assert sha(p)=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads(p.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='4',DT_TASK='TextCraft',DT_MAX_STEPS='30',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(root/'runs/textcraft-formal-stable-20261009-v1/entry'),str(dt),
 q['official_root'],str(dt/'clean/qwen35'),s['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'probe_paper_role_dt_20261010_v3.py')]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(unix=time.time(),pid=process.pid,birth=psutil.Process(process.pid).create_time(),
 devices=[4],script_sha256=HASHES['probe_paper_role_dt_20261010_v3.py'],inputs_sha256=HASHES['paper-inputs.json'],
 source_sha256=sha(p),argv=argv,physical_before=physical,host_available_before=psutil.virtual_memory().available,
 planned_DT_calls=2,original_paper_cases=4,optimizer_steps=0,rollout=0,production_changes=0,
 scope='Paper-example implementation diagnosis; current official symmetric and official clean profiles on same literal inputs and base weights. No new algorithm, numerical tolerance, training restart or parameter change.')
(out/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))

PY
