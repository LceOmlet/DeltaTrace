"""CPU audit of real pre-update IDs/masks against retained native trajectories."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
code=r'''
import hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(4)
root=Path(ROOT);source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
sys.path.insert(0,source['verl_root'])
from verl.utils.model import compute_position_id_with_mask
cap=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
records=[]
for rank in (0,1):
 path=cap/f'rank{rank}-pre-update.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
 b=payload['tensors'];a=payload['non_tensors']['dt_direct_target_artifact']
 prompt_width=b['input_ids'].shape[-1]-b['responses'].shape[-1]
 mismatches=[];mapped=0;unmapped_policy=0;policy_mismatch=0;initial_prompt_mismatch=0
 for i,artifact in enumerate(a):
  pos=torch.tensor(artifact['retained_response_positions']);kept=pos>=0
  ids=torch.tensor(artifact['response_ids']);mask=torch.tensor(artifact['policy_mask'])
  actual=b['responses'][i,kept];expected=ids[pos[kept]]
  bad=torch.where(actual!=expected)[0]
  if len(bad):mismatches.append(dict(row=i,count=len(bad),first_positions=torch.where(kept)[0][bad[:8]].tolist()))
  mapped+=int(kept.sum())
  unmapped_policy+=int(b['loss_mask'][i,-len(pos):][~kept].sum())
  policy_mismatch+=int((b['loss_mask'][i,-len(pos):][kept].bool()!=mask[pos[kept]].bool()).sum())
  prompt_ids=torch.tensor(artifact['prompt_ids'])
  initial_prompt_mismatch+=int((b['input_ids'][i,:prompt_width][-len(prompt_ids):]!=prompt_ids).sum())
 reconstructed=compute_position_id_with_mask(b['attention_mask'])
 owner_prompt=compute_position_id_with_mask(b['attention_mask'][:,:prompt_width])
 owner_response=owner_prompt[:,-1:]+torch.arange(1,b['responses'].shape[-1]+1)[None,:]
 owner_positions=torch.cat((owner_prompt,owner_response),-1)
 valid=b['attention_mask'].bool()
 records.append(dict(rank=rank,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(a),
  retained_mapped_tokens=mapped,response_ID_mismatches=mismatches,initial_prompt_mismatches=initial_prompt_mismatch,
  policy_mask_mismatches=policy_mismatch,unmapped_policy_tokens=unmapped_policy,
  owner_vs_reconstructed_valid_position_mismatches=int((owner_positions[valid]!=reconstructed[valid]).sum()),
  owner_vs_reconstructed_padding_position_mismatches=int((owner_positions[~valid]!=reconstructed[~valid]).sum()),
  CUDA_initialized=torch.cuda.is_initialized()))
print(json.dumps(dict(scope='Saved actor IDs after DT versus exact retained native trajectory IDs; original actor positions were not saved.',ranks=records)))
'''
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nROOT='+repr(transport.ROOT)+'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/'audit-original-actor-artifacts.stderr.txt').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode()
value=json.loads(result.stdout)
(HERE/'original-actor-artifact-audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
