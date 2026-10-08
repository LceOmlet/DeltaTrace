source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,json,subprocess,os
from pathlib import Path
p=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/analyze_saved_full_credit.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9ea99df401525ef75d9e51ee4818baa4fdf076595d75edb669744b3c3e6f6b04'
subprocess.run([os.environ["VENV_PYTHON"],str(p),"--records",*['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/textcraft-v3-rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/textcraft-v3-rank1.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-gdn-collection-textcraft-20261009-v1/results/rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-gdn-collection-textcraft-20261009-v1/results/rank1.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/results/rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/results/rank1.json'],"--output",'/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/full-credit-endpoint-ledger.json',*['--include-endpoint-ledger']],check=True)
q=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/full-credit-endpoint-ledger.json'); print(json.dumps(dict(remote=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())))

PY
