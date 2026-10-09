source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,time,subprocess
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
observed=[]
for pid,birth in [(1746650,1791561149.33),(1750635,1791561177.34),(1752153,1791561191.46)]:
 try:
  p=psutil.Process(pid)
  observed.append(dict(pid=pid,expected_birth=birth,current_birth=p.create_time(),same_process=p.create_time()==birth,phase=p.name()))
 except psutil.NoSuchProcess:observed.append(dict(pid=pid,expected_birth=birth,present=False))
p=psutil.Process(982372);assert p.create_time()==1791553809.84
workers=[]
for pid,birth in [(987808,1791553850.),(989860,1791553867.51)]:
 q=psutil.Process(pid);assert q.create_time()==birth
 m=q.memory_full_info()
 workers.append(dict(pid=pid,birth=birth,phase=q.name(),PSS_bytes=m.pss,RSS_bytes=m.rss,USS_bytes=m.uss))
print(json.dumps(dict(unix=time.time(),formal_pid=982372,formal_birth=p.create_time(),completed_AppWorld_debug_handles=observed,formal_workers=workers,
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15).stdout,
 host_available_bytes=psutil.virtual_memory().available,
 cgroup_memory_usage_bytes=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text().strip(),
 production_changes=0,new_model_DT_optimizer_calls=0)))
PY
