"""Package one complete real causal FA row for the unchanged existing checker.
Reference endpoint q0/k0/v0 is explicitly selected: factual v1 was not retained.
Uniform actual query starts are rebased to the checker's scalar representation.
No attention equation, tolerance, model forward, backward update or credit changes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
import torch

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
out=root/'receipts/direct-target-extreme-operator-20261007-v1/results'
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes())
env=dict(os.environ,**source['environment']);dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
paths=[str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),*source['pythonpath'].split(':'),qwen['ft_extension_root']]
sys.path[:0]=paths;env['PYTHONPATH']=':'.join(paths+[str(root/'receipts/direct-target-numerics-20261007-v4')]);env['CUDA_VISIBLE_DEVICES']='4';env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
for name in ('rank0','rank1'):
 record=json.loads((out/(name+'.json')).read_bytes());p=psutil.Process(record['pid']) if psutil.pid_exists(record['pid']) else None
 assert p is None or p.create_time()!=record['birth'],'Original worker must exit first'
file=out/'rank0-decoder27-fa.pt';d=torch.load(file,map_location='cpu',weights_only=False,mmap=True)
layout=d['layout'];starts=layout.query_starts;assert len(set(starts))==1
query_start=starts[0];sample=0;length=layout.lengths[sample];suffix=length-query_start
ops=d['ops'];attention={}
for key,name,end in [('q0','dense_q',suffix),('k0','dense_k',length),('v0','dense_v',length)]:
 value=ops[key][sample:sample+1,:,:end].transpose(1,2).contiguous()
 attention[name]=torch.cat((value,value),dim=0)
payload=dict(fa=dict(operands=dict(u=ops['u'][sample:sample+1,:,:suffix].clone()),scale=d['scale'],lengths=[length],padded_length=length,query_start=query_start,coefficient_starts=[layout.coefficient_starts[sample]]),attention_values=attention)
path=out/'actual-reference-fa27-row0.pt';assert not path.exists(),'Do not run this check twice';torch.save(payload,path);del payload,attention,d,ops
sources=root/'receipts/direct-target-numerics-20261007-v4';checker=sources/'verify_saved_fa_dtypes.py'
assert hashlib.sha256(checker.read_bytes()).hexdigest()=='7ff11d9d800a2c233b019213ae0fa9a5b603dfea51dfffabe6b9296a74aab9a9'
command=[env['VENV_PYTHON'],str(checker),'--operands',str(path),'--sources',str(sources),'--output',str(out/'fa27-official-check.json')]
record=dict(scope=__doc__,source=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()),packaged=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),sample=sample,query_start=query_start,valid_suffix=suffix,context_length=length,original_batch=4,actual_qkv_dtype='BF16',actual_upstream_dtype='FP32',endpoint='reference q0/k0/v0; not factual v1',checker=dict(path=str(checker),sha256=hashlib.sha256(checker.read_bytes()).hexdigest()),command=command,started_unix=time.time(),optimizer_steps=0,model_forwards=0,DT_calls=0)
(out/'fa27-check-packaging.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'fa27-check.stdout.txt').open('xb') as stream:r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
record.update(returncode=r.returncode,completed_unix=time.time());(out/'fa27-check-packaging.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(r.returncode)
