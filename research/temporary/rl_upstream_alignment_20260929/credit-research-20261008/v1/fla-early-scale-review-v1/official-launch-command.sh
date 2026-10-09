source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,os,hashlib,subprocess,psutil,time,re
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-early-output-scale-20261009-v1')
assert not (out/'launch.json').exists(),'Do not duplicate the bounded check'
assert hashlib.sha256((out/'check_fla_early_scale.py').read_bytes()).hexdigest()=='102bc708feecd0ec1dd0d08438c188f8190baf72cfd33b2e62d7c48f990f806f'
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 occupied'
s=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','600',env['VENV_PYTHON'],str(out/'check_fla_early_scale.py'),'--operands',str(root/'receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt'),'--precast',str(root/'receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt'),'--official',str(root/'receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py'),'--output',str(out/'official-result.json')]
with (out/'official-driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit='6af529bc29b82473c0e83fc667f65692a117f39b',script_sha256='102bc708feecd0ec1dd0d08438c188f8190baf72cfd33b2e62d7c48f990f806f',argv=argv,physical_before=physical,devices=[4],maximum_seconds=600,production_changed=False,formal_training=False)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
