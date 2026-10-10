"""Locate iteration 19 extrema in saved native actor inputs on CPU only.

No model, DT, backward, optimizer, or running worker RPC is called.
The original snapshot and source credit records remain unchanged.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RAW = HERE / 'direct-credit-records-20261009-v1'
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

audit = r'''
import hashlib,io,json,psutil,resource,time,sys
from pathlib import Path
import torch
root=Path(ROOT);formal=root/'runs/textcraft-formal-stable-20261009-v1'
assert hashlib.sha256((formal/'source.json').read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
assert psutil.Process(982372).create_time()==1791553809.84
start=time.perf_counter();workers=[]
sys.path.insert(0,str(formal/'entry'))
import counterfactual as credit_owner
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',local_files_only=True)
credit_files=[p for p in (formal/'credit-records').glob('*/*.pt')
 if 1791602257.6<int(p.stem.removeprefix('joint-'))/1e9<1791604348]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 assert psutil.Process(pid).create_time()==birth
 p=next((root/'receipts/textcraft-native-actor-incidents-20261010-v1').glob('rank*-pid'+str(pid)))/'pending-update.pt'
 blob=p.read_bytes();data=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
 assert data['update_index']==2
 batch=data['input_batch'];width=batch['responses'].shape[-1]
 mask=batch['loss_mask'][:,-width:].bool();A=batch['advantages']
 flat=int(A.masked_fill(~mask,float('inf')).reshape(-1).argmin());row,col=divmod(flat,width)
 uid=str(data['input_non_tensor_batch']['uid'][row])
 traj_uid=str(data['input_non_tensor_batch']['traj_uid'][row])
 artifact=data['input_non_tensor_batch']['dt_direct_target_artifact'][row]
 full_slot=int(artifact['retained_response_positions'][col])
 assert full_slot>=0 and artifact['response_ids'][full_slot]==int(batch['responses'][row,col])
 matches=[]
 for source_path in credit_files:
  stored=torch.load(source_path,map_location='cpu',weights_only=True)
  for source in stored['rows']:
   if source['traj_uid']!=traj_uid:continue
   slots=source['suffix_positions'];offset=int((slots==full_slot).nonzero().item())
   absolute=source['prompt_length']+offset
   assert int(source['input_ids'][absolute])==int(batch['responses'][row,col])
   d=source['source_log_ratios']
   credit=credit_owner.reward_event_token_credit(d[None,None,:],torch.tensor([[source['reward']]],dtype=torch.float32),torch.ones((1,1,d.numel()),dtype=torch.bool),source['policy_mask'][None,:],self_target_mask=source['target_mask'][None,None,:])
   assert credit.advantages.device.type=='cpu'
   matches.append(dict(path=str(source_path),sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),traj_uid=traj_uid,full_response_position=full_slot,input_position=absolute,token_id=int(source['input_ids'][absolute]),token=tokenizer.decode([int(source['input_ids'][absolute])]),context=tokenizer.decode(source['input_ids'][max(0,absolute-32):absolute+33].tolist()),reward=source['reward'],d=float(d[full_slot]),counterfactual_probability_ratio=float(torch.exp(-d[full_slot].double())),factual_target_logp=source['factual_target_logp'],joint_reference_target_logp=source['reference_target_logp'],source_token=bool(source['prior_source_mask'][full_slot]),target_token=bool(source['target_mask'][full_slot]),raw_advantage_from_original_credit_owner=float(credit.advantages[0,full_slot]),actor_saved_raw_advantage=float(batch['dt_token_advantages'][row,col]),actor_saved_Q=float(batch['dt_q_estimates'][row,col]),actor_saved_V=float(batch['dt_v_estimates'][row,col])))
 assert matches,('Missing original record for',traj_uid)
 workers.append(dict(pid=pid,rank=data['rank'],birth=birth,update_index=data['update_index'],snapshot_unix=data['unix'],path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),non_tensor_keys=list(data['input_non_tensor_batch']),batch_keys=list(batch.keys()),active_tokens=int(mask.sum()),nonfinite_counts={k:int((~torch.isfinite(batch[k])).sum()) for k in ('advantages','old_log_probs','ref_log_prob')},minimum=dict(row=row,response_position=col,uid=uid,traj_uid=traj_uid,token_id=int(batch['responses'][row,col]),advantage=float(A[row,col]),old_logp=float(batch['old_log_probs'][row,col]),ref_logp=float(batch['ref_log_prob'][row,col]),original_record_matches=matches),maximum=float(A[mask].max())))
 del data,batch,A,mask,blob
assert not torch.cuda.is_initialized()
print(json.dumps(dict(unix=time.time(),seconds=time.perf_counter()-start,workers=workers,original_DT_record_files_in_iteration=len(credit_files),credit_owner_path=credit_owner.__file__,credit_owner_sha256=hashlib.sha256(Path(credit_owner.__file__).read_bytes()).hexdigest(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,PSS_bytes=psutil.Process().memory_full_info().pss,CUDA_initialized=False,model_calls=0,DT_calls=0,optimizer_calls=0,scope='Locate saved current actor extrema and join the original token/source records. Does not measure single-deletion truth or identify NaN cause.')))
'''.replace('ROOT', repr(transport.ROOT), 1)

outer = r'''
import json,os,subprocess
from pathlib import Path
source=json.loads(Path(SOURCE).read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=source['pythonpath']
r=subprocess.run([env['VENV_PYTHON'],'-c',AUDIT],env=env,capture_output=True,text=True,timeout=45)
if r.returncode:raise RuntimeError(r.stderr)
print(r.stdout)
'''.replace('SOURCE', repr(transport.ROOT + '/runs/textcraft-formal-stable-20261009-v1/source.json'), 1).replace('AUDIT', repr(audit), 1)
command = 'source ' + transport.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + outer + '\nPY\n'
stem = RAW / 'native-input-extrema-iteration19-20261010-v4'
assert not stem.with_suffix('.json').exists()
stem.with_suffix('.command.txt').write_text(command, encoding='utf-8')
result = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(), capture_output=True, timeout=55)
stem.with_suffix('.stdout.txt').write_bytes(result.stdout)
stem.with_suffix('.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
data = json.loads(result.stdout)
stem.with_suffix('.json').write_bytes(result.stdout)
print(json.dumps(dict(saved=str(stem.with_suffix('.json')), **data), ensure_ascii=False))
