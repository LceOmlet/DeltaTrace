set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,os,subprocess
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-update-gradient-20261008-v1');source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes());env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='4,5';env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps']);dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35'];env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']]);r=subprocess.run([env['VENV_PYTHON'],str(out/'inspect_actual_token_gradient.py'),'--inspect-only'],env=env,capture_output=True);print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)

PY
