source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,json,psutil,subprocess,time
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
s=json.loads((r/'runs/textcraft-formal-stable-20261009-v1/source.json').read_bytes())
q=json.loads(Path(s['environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
p=Path(q['official_root'])/'ft_ifr_improve.py'
text=p.read_text();a=text.index('def faithfulness_test_skip_tokens(');b=text.find('\ndef ',a+4)
files=[]
for relative in ['receipts/credit-author-development-collection-20261008-v1/inspect_author_collection.py','receipts/direct-target-action-author-curve-20261007-v1/inspect_action_curve.py','receipts/direct-target-action-author-curve-20261007-v1/inspect_extreme_endpoint.py']:
 f=r/relative;files.append(dict(path=str(f),exists=f.exists(),sha256=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None))
print(json.dumps(dict(unix=time.time(),metric=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),function_source=text[a:b if b>0 else None]),helpers=files,physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host_available=psutil.virtual_memory().available)))

PY
