source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import os,json,subprocess,time,hashlib
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-memory-stable-composition-20261009-v1');source_path=r/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json';source=json.loads(source_path.read_bytes());memory=r/'candidates/direct-target-consumed-cache-release-20261007-v1';qwen=json.loads((memory/'environment.json').read_bytes())['qwen35']
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK='AppWorld',DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=str(memory/'deltatrace'),DT_ENVIRONMENT_JSON=str(memory/'environment.json'),CUDA_VISIBLE_DEVICES='-1')
old_helpers=r/'receipts/direct-target-memory-capacity-20261008-v1'
env['PYTHONPATH']=':'.join([str(out),str(old_helpers),str(memory/'deltatrace'),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(memory/'deltatrace/clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'check_appworld_native_dt_capacity.py'),'--source',str(source_path),'--failed-batch',str(r/'receipts/direct-target-native-mlp-memory-20261007-v3/failed-inputs'),'--output',str(out/'cpu-prepared-inputs-v1'),'--prepare-only']
result=subprocess.run(argv,env=env,capture_output=True,timeout=50)
(out/'prepare-v1.stdout').write_bytes(result.stdout);(out/'prepare-v1.stderr').write_bytes(result.stderr)
print(json.dumps(dict(unix=time.time(),returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace'))))
PY
