"""Describe all saved native credits; select high-impact real B4 diagnostics.
Reuses the actual credit owner; all outputs are expected-from-native diagnostics,
not newly consumed training values or parameter-gradient shares.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import torch

torch.set_num_threads(1);tick=time.time()
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/direct-target-credit-sample-20261007-v1';OUT.mkdir(exist_ok=True)
BASE=ROOT/'receipts/direct-target-prefix-runtime-20261007-v1'
EXPECTED={'textcraft':'2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52','appworld':'58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def load_owner(path):
 spec=importlib.util.spec_from_file_location('actual_credit_owner',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def summary(v):
 finite=v[torch.isfinite(v)];negative=finite[finite<0];squares=finite.double().square();total=float(squares.sum())
 return dict(count=v.numel(),nonfinite=int((~torch.isfinite(v)).sum()),negative_count=negative.numel(),nonnegative_count=int((finite>=0).sum()),min=float(finite.min()),max=float(finite.max()),sum=float(finite.double().sum()),sumsq=total,negative_sumsq=float(negative.double().square().sum()),quantiles={str(q):float(torch.quantile(finite.double(),q)) for q in (0.,.001,.01,.1,.5,.9,.99,.999,1.)},descriptive_thresholds={str(t):dict(count=int((finite<=-t).sum()),sumsq_fraction=float(finite[finite<=-t].double().square().sum())/total if total else None) for t in (1,5,10)},largest_absolute_sumsq_fraction={str(n):float(squares.topk(min(n,len(squares))).values.sum())/total if total else None for n in (1,10,100)})

tasks={}
for task,expected in EXPECTED.items():
 source_path=ROOT/f'runs/direct-target-prefix-runtime-20261007-v1/{task}/{task}-dt/source.json';assert sha(source_path)==expected
 source=json.loads(source_path.read_bytes());owner_path=Path(source['environment']['DT_ENTRY_ROOT'])/'counterfactual.py' if 'DT_ENTRY_ROOT' in source['environment'] else Path(source['pythonpath'].split(':')[0])/'counterfactual.py'
 if not owner_path.exists():owner_path=ROOT/f'candidates/direct-target-prefix-runtime-20261007-v1/{task}/entry/counterfactual.py'
 assert sha(owner_path)=='0d3412b8ac25d7c036e7a827d4bbf19a4eb2b9984a0a0956f37e82d7d86f9d07'
 owner=load_owner(owner_path);prior_values=[];policy_values=[];signed_values=[];batches=[];uids=[]
 for path in sorted((BASE/(task+'-first-dt')).glob('rank*-readout-native-batch-*.pt')):
  data=torch.load(path,map_location='cpu',weights_only=False);native=data['native_signed'];batch=dict(file=artifact(path),native_shape=list(native.shape),rows=[],negative_prior_sumsq=0.)
  for saved in sorted(data['rows'],key=lambda v:v['batch_row']):
   b=saved['batch_row'];suffix=saved['suffix_positions'];prior=saved['prior'];target=saved['target'];policy=saved['policy'];prompt=saved['prompt_length'];uids.append(saved['traj_uid'])
   d=torch.zeros(saved['width'],dtype=torch.float32);packed_positions=prompt+torch.arange(len(suffix));source_slots=suffix[prior[suffix]];packed_source=packed_positions[prior[suffix]]
   d[source_slots]=native[b,packed_source].float();reward=float(saved['row']['dt_direct_reward'])
   credit=owner.reward_event_token_credit(d[None,None,:],torch.tensor([[reward]],dtype=torch.float32),torch.ones((1,1,saved['width']),dtype=torch.bool),policy[None,:],self_target_mask=target[None,None,:]);a=credit.advantages[0]
   prior_values.append(a[prior]);policy_values.append(a[policy]);signed_values.append(native[b,packed_source]);neg=source_slots[a[source_slots]<0];positive=source_slots[a[source_slots]>0]
   def point(slot):
    i=int((suffix==slot).nonzero().item());packed=prompt+i
    return dict(response_slot=int(slot),packed_slot=packed,token_id=int(saved['selected'][packed]),original_input_slot=saved['row']['input_ids'].numel()-saved['width']+int(slot),d_FP64=float(native[b,packed]),consumed_d_FP32=float(d[slot]),expected_A_FP32=float(a[slot]),reward=reward)
   chosen={}
   if len(neg):
    ordered=neg[torch.argsort(a[neg],stable=True)];chosen['most_negative']=point(int(ordered[0]));chosen['median_negative']=point(int(ordered[len(ordered)//2]))
   if len(positive):chosen['most_positive']=point(int(positive[a[positive].argmax()]))
   sq=float(a[neg].double().square().sum());batch['negative_prior_sumsq']+=sq
   batch['rows'].append(dict(batch_row=b,traj_uid=saved['traj_uid'],trajectory_index=saved['trajectory_index'],selected_length=saved['selected'].numel(),target_offsets=saved['target_offsets'],policy_tokens=int(policy.sum()),prior_tokens=int(prior.sum()),target_tokens=int(target.sum()),negative_prior_sumsq=sq,candidates=chosen))
  batches.append(batch);del data
 all_prior=torch.cat(prior_values);all_policy=torch.cat(policy_values);all_d=torch.cat(signed_values)
 chosen=max(batches,key=lambda row:row['negative_prior_sumsq']);total_negative=float(all_prior[all_prior<0].double().square().sum())
 chosen['fraction_of_all_saved_negative_prior_sumsq']=chosen['negative_prior_sumsq']/total_negative
 chosen['selected_most_negative_sumsq_fraction']=sum(row['candidates'].get('most_negative',{}).get('expected_A_FP32',0.)**2 for row in chosen['rows'])/total_negative
 tasks[task]=dict(scope='Complete saved native captures only; AppWorld is incomplete formal DT, TextCraft complete first DT.',source=artifact(source_path),credit_owner=artifact(owner_path),native_files=len(batches),saved_trajectory_rows=len(uids),unique_traj_uids=len(set(uids)),duplicate_traj_rows=len(uids)-len(set(uids)),prior_A=summary(all_prior),policy_A=summary(all_policy),prior_native_d=summary(all_d),batches=batches,selected_batch=chosen)
# Decode only selected real positions with the already provisioned official tokenizer.
from transformers import AutoTokenizer
model=Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B');tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
for task,result in tasks.items():
 payload=torch.load(result['selected_batch']['file']['path'],map_location='cpu',weights_only=False)
 for row in result['selected_batch']['rows']:
  saved=payload['rows'][row['batch_row']]
  for point_ in row['candidates'].values():
   i=point_['packed_slot'];ids=saved['selected'];point_['decoded_token']=tokenizer.decode([point_['token_id']],skip_special_tokens=False,clean_up_tokenization_spaces=False)
   point_['context']=dict(start=max(0,i-12),end=min(len(ids),i+13),decoded=tokenizer.decode(ids[max(0,i-12):min(len(ids),i+13)].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False))
result=dict(scope=__doc__,tasks=tasks,tokenizer=dict(model=str(model),tokenizer_json=artifact(model/'tokenizer.json'),tokenizer_config=artifact(model/'tokenizer_config.json')),runtime=dict(pid=os.getpid(),unix=time.time(),seconds=time.time()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_initialized=torch.cuda.is_initialized()),limitations=['No gradient is computed; coefficient squares are not parameter-gradient shares.','Selection targets worst saved native batch by negative prior coefficient squares; it is not an unbiased error-rate sample.','AppWorld captures omit unfinished native batches; no entire-formal-batch claim.','Diagnostic thresholds1/5/10 are descriptive histogram bins, not acceptance gates.'])
assert not result['runtime']['cuda_initialized']
(OUT/'population.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(runtime=result['runtime'],tasks={task:{k:v for k,v in data.items() if k not in ('batches',)} for task,data in tasks.items()}),ensure_ascii=False))
