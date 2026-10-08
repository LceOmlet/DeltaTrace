source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,json,subprocess,os
from pathlib import Path
p=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/analyze_saved_full_credit.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='90c31b6733ec0df798295e9e3b112e75395eb1d19552aadf62d98bf03a4e37ed'
subprocess.run([os.environ["VENV_PYTHON"],str(p),"--records",*['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/textcraft-v3-rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/textcraft-v3-rank1.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-gdn-collection-textcraft-20261009-v1/results/rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-gdn-collection-textcraft-20261009-v1/results/rank1.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/results/rank0.json', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/results/rank1.json'],"--output",'/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/full-credit-bounds.json'],check=True)
q=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-mixers-collection-textcraft-20261009-v1/full-credit-bounds.json'); print(json.dumps(dict(remote=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())))

PY
