"""Recorded nonzero finite FA: original row and scalar ABI representation check.

No reference formula or tolerance is added. Calls the actual original operator
class/library; compares descriptive equality to its saved actual B4 outputs.
No model, DT, PPO, optimizer, checkpoint or formal training operation occurs.
"""
import hashlib,json,os,psutil,re,subprocess,sys,time
from pathlib import Path
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/direct-target-finite-owner-audit-20261008-v1'
source_path=ROOT/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(source_path.read_bytes());dt=Path(source['environment']['DT_ROOT'])
config=json.loads(Path(source['environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
os.environ['CUDA_VISIBLE_DEVICES']='4';os.environ.pop('MACA_VISIBLE_DEVICES',None)
sys.path[:0]=[str(dt/'clean/qwen35'),str(dt/'clean/qwen3')]
from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
import vendor_fa_finite_bf16_d256 as owner

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(owner.__file__)=='3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'
assert sha(source_path)=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 is occupied; do not overlap jobs'
assert psutil.Process(2833207).create_time()==1791370325.16
releases=[(ROOT/f'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank{i}-release-update').exists() for i in (0,1)]
assert releases==[False,False]
operand_path=ROOT/'receipts/direct-target-extreme-operator-20261007-v1/results/rank0-decoder27-fa.pt'
assert sha(operand_path)=='ed804b5cd725cd027d697b97897a076170a2b7df64bf1347cd1e6297f68bad92'
d=torch.load(operand_path,map_location='cpu',weights_only=False,mmap=True)
layout=d['layout'];starts=layout.query_starts or (layout.query_start,)*len(layout.lengths)
operation=VendorFAFiniteP1BF16D256(config['finite_library'],config['finite_library_sha256'])
OUT.mkdir(exist_ok=True);report_path=OUT/'nonzero-layout-comparison.json';assert not report_path.exists()
r=dict(scope=__doc__,unix=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),source_sha256=sha(source_path),operand_sha256=sha(operand_path),owner=dict(path=owner.__file__,sha256=sha(owner.__file__)),library=dict(path=config['finite_library'],sha256=sha(config['finite_library'])),rows=[],physical_before=physical,text_release_present=releases,model_loads=0,DT=0,optimizer_steps=0)
log=(OUT/'nonzero-layout-phases.jsonl').open('x',buffering=1)
def emit(phase,**kw):
 log.write(json.dumps(dict(phase=phase,unix=time.time(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),pss_bytes=psutil.Process().memory_full_info().pss,**kw))+'\n')
try:
 with torch.no_grad():
  for row,(length,start,coefficient) in enumerate(zip(layout.lengths,starts,layout.coefficient_starts)):
   count=length-start;ops={name:value[row:row+1,:,:length if name in ('k0','k1','v0') else count].contiguous().cuda() for name,value in d['ops'].items()}
   item=dict(row=row,length=length,query_start=start,coefficient_start=coefficient,nonzero_endpoints=dict(q=not torch.equal(ops['q0'],ops['q1']),k=not torch.equal(ops['k0'],ops['k1'])),modes={})
   for mode in ('scalar','row'):
    actual_layout=RightPaddedLengths([length],length,'cuda',coefficient_starts=[coefficient],**({'query_start':start} if mode=='scalar' else {'query_starts':[start],'query_padded_length':count}))
    emit('finite_begin',row=row,mode=mode);tick=time.perf_counter();actual=operation(ops,d['scale'],actual_layout);torch.cuda.synchronize()
    stats={}
    for name,v in actual.items():
     observed=v.cpu();expected=d['coefficients'][name][row:row+1,:,:count]
     stats[name]=dict(equal=torch.equal(observed,expected),maxabs=float((observed.float()-expected.float()).abs().max()),shape=list(v.shape),dtype=str(v.dtype),finite=bool(torch.isfinite(observed).all()))
    item['modes'][mode]=dict(seconds=time.perf_counter()-tick,checks=stats);emit('finite_complete',row=row,mode=mode);del actual,actual_layout
   r['rows'].append(item);report_path.write_text(json.dumps(r,indent=2)+'\n');del ops
 r.update(status='comparison_complete',all_saved_outputs_exact_equal=all(t['equal'] for row in r['rows'] for mode in row['modes'].values() for t in mode['checks'].values()),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),pss_bytes=psutil.Process().memory_full_info().pss,completed_unix=time.time(),precision_acceptance='No new tolerance or precision acceptance; exact comparison of the same operator and data under row/scalar representations only.')
 report_path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
finally:log.close()
