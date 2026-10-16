"""Reuse bounded raw mx-smi sampling and label the existing native worker phases."""
import hashlib
from pathlib import Path
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/sample_physical.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'ad5b15e388422430f3f2e8ed9022ca64488c2e4274a4be93d4ac718ed7120d63'
code = owner.read_text().replace("AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2'",
                                 "AUDIT=ROOT/'receipts/appworld-native-PPO-32k-speed-20261016-v3'")
code = code.replace("AUDIT/'same-session-status.json'", "AUDIT/'trial-status.json'")
anchor = "        p=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')"
replacement = '''        record['worker_phases']={}
        for pid,birth in ((3681127,1792159012.68),(3683295,1792159033.41)):
            try:
                process=psutil.Process(pid)
                if process.create_time()==birth:record['worker_phases'][str(pid)]=process.name()
            except psutil.NoSuchProcess:
                pass
'''+anchor
assert code.count(anchor) == 1
code = code.replace(anchor, replacement)
old = "        if state['status'].startswith('completed') or state['status']=='failed':break"
new = "        if state['status']=='all_native_drivers_finished' or state['status']=='failed':break"
assert code.count(old) == 1
code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())
