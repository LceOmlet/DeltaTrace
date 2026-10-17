"""Use the existing native trace reducer with current paths and eight ranks."""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-register-threads-20261017-v1'
source = AUDIT / 'source/original_summarize_formal_vllm_profile.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '9b080450b4fbc91d4ac5a6e68894e74fc224eb0be8df1370388c497d86b9b6d4'
tree = ast.parse(source.read_text())
call = next(n for n in tree.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and getattr(n.value.func, 'id', '') == 'remote')
literal = next(n.value for n in ast.walk(call)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and 'def union_seconds(' in n.value)
body = literal.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
body = body.replace('@ROOT@', str(ROOT)).replace('@RECEIPT@',
    repr(str(AUDIT / 'native-rollout-profile-one-batch-v1')))
assert body.count('for rank in (0,1):') == 1
body = body.replace('for rank in (0,1):', 'for rank in range(8):')
exec(compile(body, str(source), 'exec'), globals())
