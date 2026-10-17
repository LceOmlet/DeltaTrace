"""Bind the original protected stop/launch owner to the current v2 identities."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1/source/replace_tuning_and_launch.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '60f3897081835bfc092732b3d281ecd2315e3a2c16f0439e76f54ee665ec20c9'
code = source.read_text().replace('receipts/six-run-fresh-after-efficiency-20261017-v1',
    'receipts/six-run-fresh-after-register-threads-20261017-v1')
code = code.replace("PILOT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'",
                    "PILOT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v2'")
code = code.replace('[(3173187, 1792235540.72), (3335370, 1792237019.8), (3173204, 1792235541.21)]',
                    '[(4158912, 1792245185.89), (4158944, 1792245186.29)]')
code = code.replace('os.getsid(p.pid) == 3173204', 'os.getsid(p.pid) == 4158944')
code = code.replace('p.create_time() >= 1792235541.21', 'p.create_time() >= 1792245186.29')
code = code.replace('IDENTITIES[:2]', 'IDENTITIES[:1]').replace('IDENTITIES[2]', 'IDENTITIES[1]')
code = code.replace("AUDIT/'source/run_fresh_queue.py'", "AUDIT/'source/run_campaign.py'")
exec(compile(code, str(source), 'exec'), globals())
