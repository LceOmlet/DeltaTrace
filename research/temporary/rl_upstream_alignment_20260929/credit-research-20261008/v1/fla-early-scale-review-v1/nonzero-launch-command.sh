source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,os,hashlib,subprocess,psutil,time,re
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-early-output-scale-20261009-v1')
assert not (out/'nonzero-launch.json').exists(),'Do not duplicate this check'
assert hashlib.sha256((out/'check_fla_early_scale_nonzero.py').read_bytes()).hexdigest()=='c5badee9f41baa3dea7fb5e2381bb742ba4a459b609daf00dcd5aa16eafb285d'
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 occupied'
s=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([str(out),s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','300',env['VENV_PYTHON'],str(out/'check_fla_early_scale_nonzero.py'),'--operands',str(root/'receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt'),'--precast',str(root/'receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt'),'--official',str(root/'receipts/direct-target-numerics-20261007-v3/official-check-setup/test_gated_delta_v041.py'),'--output',str(out/'nonzero-result.json')]
with (out/'nonzero-driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit='f020afd60f354f8d636055eec647a1ce05b5eece',script_sha256='c5badee9f41baa3dea7fb5e2381bb742ba4a459b609daf00dcd5aa16eafb285d',argv=argv,devices=[4],maximum_seconds=300,production_changed=False)
(out/'nonzero-launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
