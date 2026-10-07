"""CPU description using unchanged original finite seed; no acceptance gate."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
source=json.loads(a.source.read_bytes());dt=Path(source['environment']['DT_ROOT'])
sys.path[:0]=[str(dt/'clean/qwen35'),str(dt/'clean/qwen3')]
from compiled_logprob_seed import seed_with_checks

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(inspect.getsourcefile(seed_with_checks))=='757edb728aa57e2e802ab83dc59297ec32bbbb4eb9b2ecae6ca42fea5ed1a651'
torch.set_num_threads(4);tick=time.perf_counter()
paths={m:a.out/f'rank0-{m}-head.pt' for m in ('single_EOS','original_joint_EOS')}
s,j=[torch.load(paths[m],map_location='cpu',weights_only=False) for m in paths]
assert torch.equal(s['labels'],j['labels']) and torch.equal(s['positions'],j['positions'])
n=len(j['labels']);rows=[];sums={}
for dtype in (torch.float32,torch.float64):
 result=[]
 with torch.no_grad():
  for start in range(0,n,16):
   stop=min(start+16,n);z=j['logits'][2*start:2*stop].to(dtype);zs=s['logits'][2*start:2*stop].to(dtype)
   labels=j['labels'][start:stop];seed,checks=seed_with_checks(z[0::2],z[1::2],labels);assert bool(checks)
   joint_contract=(seed.double()*(z[1::2]-z[0::2]).double()).sum(-1)
   single_contract=(seed.double()*(zs[1::2]-zs[0::2]).double()).sum(-1)
   native_joint=z.log_softmax(-1).gather(-1,labels.repeat_interleave(2)[:,None]).squeeze(-1)
   native_single=zs.log_softmax(-1).gather(-1,labels.repeat_interleave(2)[:,None]).squeeze(-1)
   for k in range(stop-start):
    result.append(dict(position=int(j['positions'][start+k]),label=int(labels[k]),native_single_effect=float(native_single[2*k+1]-native_single[2*k]),native_joint_effect=float(native_joint[2*k+1]-native_joint[2*k]),joint_seed_times_single_logit_delta=float(single_contract[k]),joint_seed_times_joint_logit_delta=float(joint_contract[k])))
   del z,zs,seed,joint_contract,single_contract,native_joint,native_single
 sums[str(dtype)]={k:sum(row[k] for row in result) for k in ('native_single_effect','native_joint_effect','joint_seed_times_single_logit_delta','joint_seed_times_joint_logit_delta')}
 if dtype==torch.float32:rows=result
coeff=j['packed_coefficients'].double();hs=s['normalized_hidden'];hj=j['normalized_hidden']
head_cross=float((coeff*(hs[1::2].double()-hs[0::2].double())).sum())
head_joint=float((coeff*(hj[1::2].double()-hj[0::2].double())).sum())
rank_records=[json.loads((a.out/f'rank{r}.json').read_bytes()) for r in (0,1)]
factual_equal=torch.equal(s['logits'][1::2],j['logits'][1::2]);hidden_equal=torch.equal(hs[1::2],hj[1::2])
head=dict(native_saved_single=float((s['logp1'].double()-s['logp0'].double()).sum()),native_saved_joint=float((j['logp1'].double()-j['logp0'].double()).sum()),joint_head_coefficient_times_single_normalized_hidden_delta=head_cross,joint_head_coefficient_times_joint_normalized_hidden_delta=head_joint,factual_logits_equal=factual_equal,factual_normalized_hidden_equal=hidden_equal,original_seed_dtype_calculation=sums,target_count=n,top_single_effect=sorted(rows,key=lambda row:abs(row['native_single_effect']),reverse=True)[:10],immediate_code_fence=[row for row in rows if row['position']==4508 and row['label']==52451])
record=dict(scope='Actual original head and decoder27 operands. Original seed function CPU FP32/FP64 description, not FA/FLA acceptance and not new credit.',source_sha256=sha(a.source),seed_owner=dict(path=inspect.getsourcefile(seed_with_checks),sha256=sha(inspect.getsourcefile(seed_with_checks))),head=head,decoder27=[dict(rank=r['rank'],**r['modes']['original_joint_EOS']['decoder27']) for r in rank_records],after_final_norm=[r['modes']['original_joint_EOS']['after_final_norm'] for r in rank_records],candidate_signed=[r['modes']['original_joint_EOS']['candidate_signed'] for r in rank_records],resources=dict(cpu_only=True,seconds=time.perf_counter()-tick),sources=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in paths.values()],limitations=['CPU FP32/FP64 seed comparisons describe actual operator operands, not a new tolerance.','The first layer sign crossing does not alone prove FA numerical error; its actual operands are retained for the existing original checker.','No production rule, gradient, reward or formal update is changed.'])
assert not torch.cuda.is_initialized()
(a.out/'operator-analysis.json').write_text(json.dumps(record,indent=2)+'\n');(a.out/'head-target-effects.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(record))
