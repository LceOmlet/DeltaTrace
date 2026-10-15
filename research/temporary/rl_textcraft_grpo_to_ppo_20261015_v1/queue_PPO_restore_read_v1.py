"""Use the existing native Ray state reader after PPO migration, without model calls."""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/queue_native_PPO_restore_read_v55.py'
code = source.read_text()
changes = {
    "else 'v55'": "else 'v117'",
    "('v55','v56')": "('v117',)",
    "version=='v56'": "version=='v117'",
    'native-PPO-resume-after-v47-v55-job.json': 'native-PPO-resume-GPU45-v117-job.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
