set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,hashlib,ast
from pathlib import Path
p=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')/'candidates/direct-target-prefix-runtime-20261007-v1/appworld/entry/deltatrace_rollout.py'
s=p.read_text();tree=ast.parse(s);methods={}
for n in ast.walk(tree):
 if isinstance(n,ast.ClassDef) and n.name=='DeltaTraceRolloutProducer':
  for f in n.body:
   if isinstance(f,ast.FunctionDef) and f.name=='__init__':methods[f.name]=ast.get_source_segment(s,f)
imports=[ast.get_source_segment(s,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),imports=imports,methods=methods,scope='Read-only actual producer constructor ownership audit; no import/model/DT/update')))

PY
