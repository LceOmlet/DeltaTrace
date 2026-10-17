"""Use the original physical dmon/phase reducer for the fresh v3 identities."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1/source/summarize_fresh_utilization.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '1e502763c2ebc793b7e4e47ff03f5272c9f1da154c0ec6f711156f4551135433'
code = source.read_text().replace('six-run-fresh-after-efficiency-20261017-v1',
                                'six-run-fresh-after-register-threads-20261017-v1')
code = code.replace('eight-gpu-sequential-20261017-v2', 'eight-gpu-sequential-20261017-v3')
code = code.replace('-8gpu-200-20261017-v2', '-8gpu-200-20261017-v3')
exec(compile(code, str(source), 'exec'), globals())
