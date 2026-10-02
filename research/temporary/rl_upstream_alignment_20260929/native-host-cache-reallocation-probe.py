import json,os,subprocess,time,re
from pathlib import Path
raw=subprocess.check_output(['mx-smi'],text=True)
lines=raw.splitlines();gpu_line=next(i for i,s in enumerate(lines) if re.match(r'\|\s+6\s+MetaX C550\s*\|',s))
used=int(re.search(r'(\d+)/65536 MiB',lines[gpu_line+1]).group(1))
assert '0%' in lines[gpu_line] and used<2048,('GPU6 must be idle for this bounded native allocator probe',lines[gpu_line:gpu_line+2])
import torch
init_started=time.perf_counter();torch.cuda.init();cuda_init_s=time.perf_counter()-init_started
assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
size=256*1024*1024
# Native CPU pinned allocation only. No model, forward, gradient, generation,
# or device tensor is created. Maximum live probe storage is 256MiB.
stats=lambda:dict(torch.cuda.memory.host_memory_stats())
start=time.perf_counter();p=torch.empty(size,dtype=torch.uint8,pin_memory=True);initial=time.perf_counter()-start
p[0]=17;p[-1]=31;assert int(p[0])==17 and int(p[-1])==31
del p
samples=[]
for i in range(4):
 before=stats();t=time.perf_counter();p=torch.empty(size,dtype=torch.uint8,pin_memory=True);warm=time.perf_counter()-t
 del p;t=time.perf_counter();torch._C._host_emptyCache();release=time.perf_counter()-t
 after_release=stats();t=time.perf_counter();p=torch.empty(size,dtype=torch.uint8,pin_memory=True);cold=time.perf_counter()-t
 p[0]=17;p[-1]=31;assert int(p[0])==17 and int(p[-1])==31
 after=stats();del p
 samples.append(dict(index=i,warm_allocation_s=warm,release_s=release,reallocation_s=cold,
  before={k:before[k] for k in ('reserved_bytes.current','num_host_alloc','num_host_free')},
  after_release={k:after_release[k] for k in ('reserved_bytes.current','num_host_alloc','num_host_free')},
  after_reallocation={k:after[k] for k in ('reserved_bytes.current','num_host_alloc','num_host_free')}))
torch._C._host_emptyCache()
r=dict(unix=time.time(),pid=os.getpid(),torch_version=str(torch.__version__),gpu_before=raw,
 scope='Bounded native allocator interface cost only; not a training speed comparison or new numerical tolerance',
 maximum_live_bytes=size,cuda_init_s=cuda_init_s,initial_allocation_s=initial,samples=samples)
path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')/'receipts/host-memory-20261002'/f"native-host-reallocation-{int(time.time())}.json"
path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(receipt=str(path),**r)))
