"""Recover raw-credit moments and infer mask count with a saved-batch calibration."""
import importlib.util
import json
import math
from pathlib import Path
import time

import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
source=json.loads((RUN/'source.json').read_bytes())
spec=importlib.util.spec_from_file_location('bound_qv_owner',Path(source['entry'])/'counterfactual.py')
owner=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=owner
spec.loader.exec_module(owner)
raw=json.loads((OUT/'temporal-native-reports-v1.json').read_bytes())
work=json.loads((OUT/'native-workload-temporal-v1.json').read_bytes())['rows']
file_groups=[]
for log in sorted(raw['raw_reports']):
    groups,pending=[],[]
    for line in Path(log).read_text(errors='replace').splitlines():
        if '[DT direct joint record] ' in line:
            pending.append(json.loads(line.split('[DT direct joint record] ',1)[1])['path'])
        if '[DeltaTrace readout] ' in line:
            obj=json.loads(line.split('[DeltaTrace readout] ',1)[1])
            if 'raw_advantage_groups' in obj:
                groups.append(pending);pending=[]
    assert len(groups)==85,(log,len(groups))
    file_groups.append(groups)
receipts=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
full=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
mask=full.batch['response_mask'].bool()
native_a=full.batch['dt_token_advantages']
native_valid=native_a[mask].double()
native_count=int(mask.sum())
native_calibration={'retained_policy_count':native_count,'raw_sum':float(native_valid.sum()),
    'raw_sumsq':float(native_valid.square().sum()),'whitened_target_coefficient':float(full.batch['advantages'][mask].max())}
result={'started_unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'rows':[],
    'scope_limits':[
        'This infers a mask count and whitening moments from the native rounded maximum and existing raw credits; not an exact historical update reconstruction.',
        'Iterations with success histories beyond the native10240 response cap are excluded from the inversion, rather than silently using unretained credits.',
        'Mask count intervals below account for three-decimal maximum rounding; floating-point and reconstruction error are independently calibrated at iteration81, not declared universally zero.',
        'Native B4 row placement and historical parameter-gradient directions are still not recovered by this inference.',
    ],'snapshot81_native_calibration':native_calibration}

def alpha(n,s,ss):
    return (1-s/n)/math.sqrt((ss-s*s/n)/(n-1)+1e-8)

def inverse(value,low,high,s,ss):
    if not alpha(low,s,ss)<=value<=alpha(high,s,ss):return None
    for _ in range(80):
        mid=(low+high)/2
        if alpha(mid,s,ss)<value:low=mid
        else:high=mid
    return (low+high)/2

for step in range(1,86):
    rows={};paths=[]
    for groups in file_groups:
        for f in groups[step-1]:
            paths.append(f)
            payload=torch.load(f,map_location='cpu',weights_only=False)
            for r in payload['rows']:
                rows.setdefault(r['traj_uid'],r)
    assert len(rows)==work[step-1]['nonzero_requests'],(step,len(rows),work[step-1])
    s,ss,count,target_count=0.,0.,0,0
    too_long=[]
    snapshot_diffs=[]
    for uid,row in rows.items():
        policy,target=row['policy_mask'].bool(),row['target_mask'].bool()
        d=row['source_log_ratios']
        a=owner.reward_event_token_credit(d[None,None,:],torch.tensor([[row['reward']]],dtype=torch.float32),
            torch.ones((1,1,len(d)),dtype=torch.bool),policy[None,:],self_target_mask=target[None,None,:]).advantages[0]
        v=a[policy].double()
        s+=float(v.sum());ss+=float(v.square().sum());count+=v.numel();target_count+=int(target.sum())
        if len(row['input_ids'])-row['prompt_length']>10240:too_long.append(uid)
        if step==81:
            indexes=[i for i,k in enumerate(full.non_tensor_batch['traj_uid']) if str(k)==uid]
            assert len(indexes)==1
            i=indexes[0]
            art=full.non_tensor_batch['dt_direct_target_artifact'][i]
            assert torch.equal(torch.tensor(art['response_ids']),row['input_ids'][row['prompt_length']:])
            positions=torch.as_tensor(art['retained_response_positions']);keep=positions>=0
            snapshot_diffs.append(float((native_a[i,keep]-a[positions[keep]]).abs().max()))
    maximum=raw['metrics'][step-1]['values']['critic/advantages/max']
    row={'step':step,'success_count':len(rows),'success_policy_tokens':count,
        'success_target_tokens':target_count,'success_raw_sum':s,'success_raw_sumsq':ss,
        'full_history_policy_tokens':work[step-1]['full_history_policy_tokens'],
        'native_whitened_maximum_rounded':maximum,'truncated_successes':too_long,
        'native_grad_norm':raw['metrics'][step-1]['values']['actor/grad_norm'],
        'raw_file_count':len(paths),'raw_files':paths}
    if not too_long:
        low=float(count);high=float(work[step-1]['full_history_policy_tokens'])
        values=[alpha(low+(high-low)*j/20,s,ss) for j in range(21)]
        assert all(a<=b for a,b in zip(values,values[1:])),(step,values)
        n=inverse(maximum,low,high,s,ss)
        interval=[inverse(maximum-.0005,low,high,s,ss),inverse(maximum+.0005,low,high,s,ss)]
        row.update(inferred_retained_policy_count=n,inferred_count_rounding_interval=interval)
        if n is not None:
            mean=s/n;std=math.sqrt((ss-s*s/n)/(n-1)+1e-8)
            row.update(inferred_raw_mean=mean,inferred_raw_std=std,
                inferred_zero_reward_whitened_coefficient=-mean/std,
                inferred_success_target_fraction=target_count/n)
    if step==81:
        row['snapshot81_native_calibration']={
            'raw_sum_difference':s-native_calibration['raw_sum'],
            'raw_sumsq_difference':ss-native_calibration['raw_sumsq'],
            'passive_to_native_raw_credit_max_abs_difference':max(snapshot_diffs),
            'predicted_maximum_at_native_count':alpha(native_count,s,ss),
            'inferred_count_error':None if row.get('inferred_retained_policy_count') is None else row['inferred_retained_policy_count']-native_count,
        }
    result['rows'].append(row)
    if step%10==0:
        (OUT/'whitening-inference-progress-v1.json').write_text(json.dumps({'completed_iterations':step,'unix':time.time(),'model_calls':0})+'\n')
result['finished_unix']=time.time()
assert not torch.cuda.is_initialized()
dest=OUT/'whitening-temporal-inference-v1.json'
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':result}))
