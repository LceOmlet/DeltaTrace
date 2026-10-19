"""Read the existing native dmon/phase observations for this queue only."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/report_native_utilization.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='d87376e903e5827025efb33d464b18d8df78dd31c4fa68cb438489a10595706d'
code=owner.read_text().replace('six-run-fresh-validated-efficiency-20261018-v1','dtpo-v9-clip-u-postgroup-variance-20261019-v2')
code=code.replace('eight-gpu-sequential-20261018-v5','eight-gpu-v9-clip-u-20261019-v2')
code=code.replace('-8gpu-200-20261018-v5','-8gpu-200-20261019-v2')
exec(compile(code,str(owner),'exec'),globals())
