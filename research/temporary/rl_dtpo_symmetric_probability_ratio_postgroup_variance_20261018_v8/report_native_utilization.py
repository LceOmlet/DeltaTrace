"""Use the existing physical dmon and phase observations for this queue only."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/report_native_utilization.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d87376e903e5827025efb33d464b18d8df78dd31c4fa68cb438489a10595706d'
code=source.read_text().replace('six-run-fresh-validated-efficiency-20261018-v1','dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8')
code=code.replace('eight-gpu-sequential-20261018-v5','eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8')
code=code.replace('-8gpu-200-20261018-v5','-8gpu-200-20261018-v8')
exec(compile(code,str(source),'exec'),globals())
