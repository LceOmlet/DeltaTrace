/opt/conda/bin/python - <<'PY'

import hashlib,json,re,time,psutil
from pathlib import Path
p=psutil.Process(3592468);assert p.create_time()==1791291997.14
rows=[]
for name in ['/tmp/ray/session_2026-10-06_21-06-57_551176_3592468/logs/worker-bf529f28a9b1d62fdd0ae9b7797e83a8c78eee8aec0e467dc1a4bd00-01000000-3598938.out', '/tmp/ray/session_2026-10-06_21-06-57_551176_3592468/logs/worker-4d317509ed99e3b3841f013a76b42845bc625f8c3dc5c8d0841aaac9-01000000-3601412.out']:
 path=Path(name);size=path.stat().st_size;digest=hashlib.sha256();groups=[];batches=[];plans=[]
 with path.open('rb') as f:
  data=f.read(size)
 digest.update(data)
 for line in data.decode(errors='replace').splitlines():
  if line.startswith('[DeltaTrace readout] '):
   d=json.loads(line[len('[DeltaTrace readout] '):]);tr=d.get('traces',[])
   by_batch={}
   for x in tr:by_batch.setdefault(x['owner_batch_index'],[]).append(x)
   groups.append({k:v for k,v in d.items() if k not in ('traces','minimum_log_ratio_batch','actual_row_lengths')})
   groups[-1]['actual_row_lengths_summary']={'count':len(d.get('actual_row_lengths',[])),'sum':sum(d.get('actual_row_lengths',[])),'max':max(d.get('actual_row_lengths',[0]))}
   groups[-1]['trace_summary']={'count':len(tr),'context_tokens_sum':sum(x['context_tokens'] for x in tr),'compute_token_slots':sum(x['compute_tokens'] for x in tr),'batch_count':len(by_batch),'max_context':max([x['context_tokens'] for x in tr] or [0])}
   groups[-1]['preceding_logged_batches']=batches;batches=[]
  elif line.startswith('[DT EOS minibatch] '):
   m=re.search(r'contrasts=(\d+) length=(\d+) seconds=([^ ]+) batch=(\d+)/(\d+) d_min=([^ ]+) d_max=([^ ]+)',line)
   if m:batches.append(dict(contrasts=int(m[1]),length=int(m[2]),seconds=float(m[3]),index=int(m[4]),total=int(m[5]),d_min=float(m[6]),d_max=float(m[7])))
  elif line.startswith('[DT EOS plan] '):plans.append(line)
 rows.append(dict(source=name,source_prefix_bytes=size,source_prefix_sha256=digest.hexdigest(),groups=groups,pending_logged_batches=batches,plans=plans))
print(json.dumps(dict(observed_unix=time.time(),scope='CPU-only original actor structured-report extraction. Zero RPC, model calls, new profile, changes or signals.',driver_pid=3592468,driver_birth=p.create_time(),records=rows)))

PY
