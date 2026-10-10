source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
REMOTE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-role-author-curves-20261010-v1'
DIGEST='b2136d4594391e8f7326e73254e8a19e3440b21e179383d156d2e8bf0e135c82'

import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(REMOTE)
assert not (out/'launch.json').exists()
assert hashlib.sha256((out/'probe_paper_author_curves_20261010.py').read_bytes()).hexdigest()==DIGEST
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 occupied'
source=root/'runs/textcraft-formal-stable-20261009-v1/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads(source.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='4',DT_TASK='TextCraft',DT_MAX_STEPS='30',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(root/'receipts/credit-author-development-collection-20261008-v1'),
 str(root/'receipts/direct-target-action-author-curve-20261007-v1'),
 str(root/'runs/textcraft-formal-stable-20261009-v1/entry'),str(dt),q['official_root'],
 str(dt/'clean/qwen35'),s['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'probe_paper_author_curves_20261010.py')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(unix=time.time(),pid=p.pid,birth=psutil.Process(p.pid).create_time(),script_sha256=DIGEST,
 devices=[4],argv=argv,physical_before=physical,host_available_before=psutil.virtual_memory().available,
 planned_native_forward_calls=63,DT_calls=0,optimizer_steps=0,production_changes=0,
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(out/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))

PY
