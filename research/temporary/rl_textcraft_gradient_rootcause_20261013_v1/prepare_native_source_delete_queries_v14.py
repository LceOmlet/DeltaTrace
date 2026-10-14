"""Bind the already observed Action/source forks to their original DT requests.

This prepares a finite diagnostic of the disputed source values. It does not
select new training data, define a parser, or replace any training credit.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
from verl import DataProto
from reward_readout import DirectActionTargetReadout

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-source-delete-values-v14'
DEST.mkdir(exist_ok=True)
assert not (DEST/'query-manifest.json').exists(),'Preserve a prepared diagnostic.'
forks=json.loads((OUT/'native-QV-Action-direction-v12.json').read_bytes())['groups']
sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
raw_paths=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows'][80]['raw_files']
raw={}
for p in raw_paths:
    for i,r in enumerate(torch.load(p,map_location='cpu',weights_only=False)['rows']):
        raw.setdefault(r['traj_uid'],(r,p,i))
points={}
for group in forks:
    for point in group['rows']:
        if point['target'] or point['reward']==0:continue
        i,j=point['row'],point['position'];uid=str(data.non_tensor_batch['traj_uid'][i])
        artifact=data.non_tensor_batch['dt_direct_target_artifact'][i]
        response_slot=int(artifact['retained_response_positions'][j])
        assert response_slot>=0
        row,path,index=raw[uid]
        ids=row['input_ids'];prompt=int(row['prompt_length']);response=ids[prompt:]
        policy=row['policy_mask'][:len(response)].bool();target=row['target_mask'][:len(response)].bool()
        assert policy[response_slot] and not target[response_slot]
        assert int(response[response_slot])==point['token_id']
        prepared=DirectActionTargetReadout._prepare_row(dict(input_ids=ids,
            attention_mask=torch.ones_like(ids),responses=response,policy_mask=policy,
            target_mask=target,dt_direct_reward=row['reward'],traj_uid=uid),i)
        locations=prepared['suffix_positions'].eq(response_slot).nonzero().flatten()
        assert len(locations)==1
        packed_slot=prepared['prompt_length']+int(locations[0])
        d=float(row['source_log_ratios'][response_slot])
        key=(uid,response_slot)
        if key not in points:
            points[key]=dict(UID=uid,original_native_row=i,native_response_slot=j,
                original_response_slot=response_slot,packed_slot=packed_slot,token_id=point['token_id'],
                saved_DT_d=d,saved_V=point['V'],saved_Q=point['Q'],saved_raw_A=point['raw_A'],
                original_raw_file=path,original_raw_row=index,
                selected=prepared['selected'],case=prepared['case'],target_offsets=prepared['target_offsets'])
        assert float(data.batch['dt_v_estimates'][i,j])==point['V']
assert points,'No observed source forks to measure.'
queries=list(points.values())
assert all(len(p['selected'])<=32768 for p in queries)
payload=DEST/'queries.pt';torch.save(queries,payload)
owner=Path(inspect.getsourcefile(DirectActionTargetReadout))
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),iteration=81,checkpoint=80,
    query_file=str(payload),query_SHA256=hashlib.sha256(payload.read_bytes()).hexdigest(),
    distinct_source_points=len(queries),original_same_prefix_groups=len(forks),
    lengths=[len(p['selected']) for p in queries],
    selection='Every nonzero-reward non-target occurrence from all14 previously measured within-B4 Action/source fork groups, deduplicated by original UID/response slot. No new extremum selection.',
    preparation_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest()),
    question='Do the saved DT source V values that favored immediate Action agree in sign and scale with native single-EOS deletion of those exact sources, at the actual preceding checkpoint80 and the same saved joint Y?',
    why_needed='The observed negative-V term favors Action in12 of14 finite groups. The saved85-batch inventory includes rare opposite Action_source samples. Neither proves whether a practical DT ratio error drives this preference. Earlier native references are from different runs/weights/UIDs.',
    decision='A large native-versus-DT discrepancy would locate a practical ratio-estimation contribution; comparable source V would retain the target/source baseline construction as the relevant mechanism. Neither constitutes population bias proof, a fresh-training prediction, or repair acceptance.',
    operations=dict(model=0,DT=0,backward=0,optimizer=0,rollout=0),
    formal_training_changed=False,accepted_repair=None,deployed_repair=None)
assert not torch.cuda.is_initialized()
dest=DEST/'query-manifest.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
