"""Record physical devices and protected identities before the bounded run."""
import json
from pathlib import Path
import re
import subprocess
import time
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-native-ppo-speed-20261016-v1'
smi = subprocess.check_output(['mx-smi'], text=True)
occupied = [line for line in smi.splitlines() if re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+', line)]
assert not occupied, occupied
protected = []
for pid, birth in ((878971,1792091830.03),(3946004,1792041120.19)):
    p = psutil.Process(pid)
    assert p.create_time() == birth
    protected.append(dict(pid=pid,birth=birth,workers=[dict(pid=c.pid,birth=c.create_time(),phase=c.name())
        for c in p.children(recursive=True) if 'WorkerDict' in c.name()]))
cgroup = Path('/sys/fs/cgroup/memory')
memory = {name:(cgroup / name).read_text() for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt','memory.stat')}
record = dict(collected_unix=time.time(), physical_gpus=[2,3], mx_smi=smi, protected=protected,
              memory=memory, host_memory=dict(psutil.virtual_memory()._asdict()),
              disk=subprocess.check_output(['df','-B1',str(ROOT)],text=True))
(AUDIT / 'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(physical_gpus=[2,3],occupied=occupied,protected=protected,
    cgroup_usage_GiB=int(memory['memory.usage_in_bytes'])/2**30,
    cgroup_limit_GiB=int(memory['memory.limit_in_bytes'])/2**30,
    host_available_GiB=record['host_memory']['available']/2**30,disk=record['disk'])))
