"""Replay bound Q/V/A composition from actual recorded DT outputs into saved PPO slots.

This checks the DTPO adaptation boundary, not DT attribution accuracy or a new
credit method. It calls the original composition owner and maps unchanged IDs.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
source=json.loads((RUN/'source.json').read_bytes())
path=Path(source['entry'])/'counterfactual.py'
spec=importlib.util.spec_from_file_location('original_transport_composition_v25',path)
owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
receipts=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
raw_paths=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows'][80]['raw_files']
saved=defaultdict(list)
for raw_path in raw_paths:
    for i,row in enumerate(torch.load(raw_path,map_location='cpu',weights_only=False)['rows']):
        saved[str(row['traj_uid'])].append((raw_path,i,row))
rows=[]
keys=dict(dt_token_advantages='advantages',dt_q_estimates='q_estimates',dt_v_estimates='v_estimates')
for i,uid in enumerate(data.non_tensor_batch['traj_uid']):
    reward=float(data.batch['token_level_rewards'][i].sum())
    if reward==0:
        rows.append(dict(row=i,UID=str(uid),reward=reward,zero_reward_native_max_abs={k:float(data.batch[k][i].abs().max()) for k in keys}));continue
    artifact=data.non_tensor_batch['dt_direct_target_artifact'][i]
    positions=torch.as_tensor(artifact['retained_response_positions'])
    candidates=[]
    for raw_path,raw_index,row in saved.get(str(uid),[]):
        d=row['source_log_ratios'];policy=row['policy_mask'].bool();target=row['target_mask'].bool()
        keep=(positions>=0)&(positions<len(d))
        recorded_ids=row['input_ids'][row['prompt_length']:]
        identity_keep=keep&(positions<len(recorded_ids))
        tokens_equal=torch.equal(data.batch['responses'][i,identity_keep],recorded_ids[positions[identity_keep]])
        values=owner.reward_event_token_credit(d[None,None,:],torch.tensor([[row['reward']]],dtype=torch.float32),
            torch.ones((1,1,len(d)),dtype=torch.bool),policy[None,:],self_target_mask=target[None,None,:])
        differences={};conflicts={}
        for key,name in keys.items():
            expected=getattr(values,name)[0,positions[keep]].float()
            actual=data.batch[key][i,keep].float()
            difference=(expected-actual).abs()
            differences[key]=float(difference.max()) if len(difference) else None
            conflicts[key]=int(difference.ne(0).sum())
        candidates.append(dict(raw_path=raw_path,raw_row=raw_index,
            reward_equal=reward==float(row['reward']),native_response_IDs_equal=tokens_equal,
            covered_native_positions=int(keep.sum()),retained_positions_outside_record=int(((positions>=0)&~keep).sum()),
            differences=differences,nonzero_difference_counts=conflicts))
    rows.append(dict(row=i,UID=str(uid),reward=reward,raw_candidates=candidates))
report=dict(unix=time.time(),scope=__doc__,rows=rows,
    composition_owner=dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest()),
    source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limits=['Every candidate raw record is reported; duplicated padded UID records are not silently selected by closeness.',
        'Only positions actually covered by a recorded vector are compared. Missing/synthetic positions are reported and not imputed as a successful equality.',
        'An equality checks representation/composition transport only, not causal-value accuracy, model gradient correctness or the complete85-iteration root cause.'],
    GPU_model_calls=0,new_DT_calls=0,new_optimizer_updates=0,rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
assert not torch.cuda.is_initialized()
dest=OUT/'DT-to-PPO-transport-audit-v25.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
