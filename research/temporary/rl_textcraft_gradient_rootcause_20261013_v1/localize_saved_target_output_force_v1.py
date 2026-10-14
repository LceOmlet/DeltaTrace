"""Localize the successful-target score signal in original iteration81 tensors."""
from collections import defaultdict
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
src=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(s['input']) for s in src])
mask=data.batch['response_mask'].bool()
rows=[];tokens=defaultdict(lambda:{'count':0,'selected_probability_deficit_weight_sum':0.,'selected_probability_deficit_weight_sumsq':0.})
events=[]
for i,art in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
    if float(data.batch['token_level_rewards'][i].sum())==0:continue
    positions=torch.as_tensor(art['retained_response_positions'])
    target=torch.zeros_like(mask[i]);keep=positions>=0
    target[keep]=torch.as_tensor(art['target_mask'],dtype=torch.bool)[positions[keep]]
    lp=data.batch['old_log_probs'][i].double()
    a=data.batch['advantages'][i].double()
    start=(i//4)*4
    n=int(mask[start:start+4].sum())
    c=a[target]/(16*n)
    deficit=-lp[target].expm1()
    out=c*deficit
    selected=target.nonzero().flatten().tolist()
    row={'row':i,'native_UID':str(data.non_tensor_batch['traj_uid'][i]),'mini':(i%128)//32,
        'B4_policy_count':n,'policy_count':int(mask[i].sum()),'target_count':len(selected),
        'target_joint_NLL':float(-lp[target].sum()),'target_logit_L1':2*float(out.abs().sum()),
        'target_logit_L2_lower':float(out.square().sum().sqrt()),
        'minimum_selected_probability':float(lp[target].exp().min())}
    rows.append(row)
    for j,pos in enumerate(selected):
        token_id=int(data.batch['responses'][i,pos])
        key=(row['mini'],token_id)
        r=tokens[key];r['count']+=1
        r['selected_probability_deficit_weight_sum']+=abs(float(out[j]))
        r['selected_probability_deficit_weight_sumsq']+=float(out[j])**2
        events.append({'row':i,'mini':row['mini'],'response_position':pos,'token_id':token_id,
            'token_text':tok.decode([token_id]),'old_logp':float(lp[pos]),
            'probability_deficit_weight':float(out[j]),'B4_policy_count':n,
            'prefix_tail':tok.decode(data.batch['responses'][i,max(0,pos-55):pos].tolist(),skip_special_tokens=False),
            'following':tok.decode(data.batch['responses'][i,pos:pos+15].tolist(),skip_special_tokens=False)})
by_token=[]
for (mini,token_id),record in tokens.items():
    by_token.append({'mini':mini,'token_id':token_id,'token_text':tok.decode([token_id]),**record})
report={'scope':__doc__,'model_GPU_calls':0,'rows':rows,
    'top_tokens_each_mini':{str(m):sorted([r for r in by_token if r['mini']==m],key=lambda r:r['selected_probability_deficit_weight_sumsq'],reverse=True)[:20] for m in range(4)},
    'top_occurrences_each_mini':{str(m):sorted([r for r in events if r['mini']==m],key=lambda r:abs(r['probability_deficit_weight']),reverse=True)[:16] for m in range(4)},
    'limits':['Output-logit force is a native old-policy localization; parameter Jacobians and cross-token directions are not replaced by these scores.',
        'The token ratio is at saved old-policy. Later optimizer minibatches change actor weights in the original training.']}
assert not torch.cuda.is_initialized()
path=OUT/'saved-target-output-force-v1.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
