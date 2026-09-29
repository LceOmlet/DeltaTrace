"""Honor the user's environment-only scope; stop only the recorded LOOP probes."""
import json
from pathlib import Path
import psutil
import time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit=root/'receipts/upstream-alignment-20260929'
targets=[]
for pid in (2457774, 2457779):
    try:
        p=psutil.Process(pid)
        command=' '.join(p.cmdline())
        if not any(str(audit/name) in command for name in ('verify_loop_model_boundary.py','verify_loop_vllm_boundary.py')):
            raise RuntimeError(f'Refusing a changed process identity: {pid}')
        targets.extend(p.children(recursive=True)+[p])
    except psutil.NoSuchProcess:
        pass
unique={p.pid:p for p in targets}
for p in reversed(list(unique.values())):
    try: p.terminate()
    except psutil.NoSuchProcess: pass
gone,alive=psutil.wait_procs(list(unique.values()),timeout=5)
for p in alive:
    if p.status()!=psutil.STATUS_ZOMBIE: p.kill()
record=dict(reason='User: do not run original LOOP; extract environment/config only and compare with project.',
    stopped_pids=list(unique), unix_time=time.time())
(audit/'loop-training-probes-stopped.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
