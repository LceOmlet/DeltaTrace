set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=''
"$VENV_PYTHON" - <<'PY'

import hashlib,json,os,time,torch
from pathlib import Path
from transformers import AutoTokenizer
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
assert os.environ['CUDA_VISIBLE_DEVICES']==''
tokenizer=AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',local_files_only=True)
cases=[('textcraft',21,666,'19b18c5f4204ec4d88e40d72d350a06954f64452824a8662e95497319f89a37a'),
       ('appworld',16,2883,'3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a')]
out=dict(unix=time.time(),cuda_initialized=torch.cuda.is_initialized(),tasks={})
for task,batch,position,expected in cases:
 p=root/('receipts/direct-target-prefix-runtime-20261007-v1/'+task+'-first-dt/rank1-readout-native-batch-'+str(batch)+'.pt')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 value=torch.load(p,map_location='cpu',weights_only=False)
 selected=next(r for r in value['rows'] if r['batch_row']==3)
 ids=selected['selected'].tolist();case=selected['case'];targets=[case['prompt_length']+j for j in selected['target_offsets']]
 relevant=[j for j in targets if position-20<=j<position+100]
 out['tasks'][task]=dict(path=str(p),sha256=expected,traj_uid=selected['traj_uid'],
     selected_keys=list(selected),original_row_keys=list(selected['row']),
     token_id=ids[position],token=tokenizer.decode([ids[position]]),packed_position=position,
     context=dict(start=position-20,end=position+100,text=tokenizer.decode(ids[position-20:position+100])),
     nearby_target_positions=relevant,nearby_target_tokens=[dict(position=j,id=ids[j],text=tokenizer.decode([ids[j]])) for j in relevant],
     original_row_metadata={k:v for k,v in selected['row'].items() if k not in ('input_ids','attention_mask','responses','response_ids','policy_mask','target_mask','prompt_ids','retained_response_positions') and not torch.is_tensor(v) and isinstance(v,(str,int,float,bool,type(None)))})
 if task=='textcraft':
  import psutil
  hits=[]
  hashes=['1563ce74f298769893b360398fd1bacf16c376439b1d462e56ef9dc6f905be59','a1970da0bbf462b203314cbf29cc8ecd8c81e526fe1b6ee944978a76bbadd34e']
  for rank,expected_actor in enumerate(hashes):
   held=root/('receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank'+str(rank)+'-pre-update.pt')
   assert hashlib.sha256(held.read_bytes()).hexdigest()==expected_actor
   saved=torch.load(held,map_location='cpu',weights_only=False)
   indices=[i for i,u in enumerate(saved['non_tensors']['traj_uid']) if str(u)==selected['traj_uid']]
   for index in indices:
    artifact=saved['non_tensors']['dt_direct_target_artifact'][index]
    hits.append(dict(path=str(held),bytes=held.stat().st_size,sha256=expected_actor,
       actor_rank=rank,local_row=index,entries=artifact['payload_metadata'][:2],
       pss_bytes=psutil.Process().memory_full_info().pss))
   del saved
  assert hits
  assert all(h['entries']==hits[0]['entries'] for h in hits)
  out['tasks'][task]['original_post_execution_metadata']=dict(hits=hits,
      alignment='Original traj_uid, not a presumed DT/actor rank correspondence')
assert not torch.cuda.is_initialized()
print(json.dumps(out,ensure_ascii=False))

PY
