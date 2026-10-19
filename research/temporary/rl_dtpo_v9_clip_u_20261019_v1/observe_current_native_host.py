"""Resume the original append-only host/PSS/XTT and native-phase observer."""
import hashlib
from pathlib import Path
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/observe_campaign.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '1b8d77694f178bf12f6fb3b0fb297a55566b253f04c03fe3697434b3c93c8388'
assert sys.argv[1:] == ['--resources-watch']
exec(compile(owner.read_text(),str(owner),'exec'),globals())
