"""Prepare exact positive-row derivative carriers, never historical full batches.

Native positive IDs, masks and original placement are retained. Failure rows
are explicitly derivative-zero placeholders copied from iteration81. Only
offline diagnostic coefficient transport compensates for their denominator;
it is never a training/credit implementation or a proposed fix.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'historical-target-derivatives-v1'
DEST.mkdir(exist_ok=True)
src=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
prototype=DataProto.concat([DataProto.load_from_disk(s['input']) for s in src])
selected=torch.load(OUT/'historical-success-ID-placement-v1.pt',map_location='cpu',weights_only=False)
moments=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows']
width=prototype.batch['responses'].shape[1]
prompt_width=prototype.batch['prompts'].shape[1]
inactive=~prototype.batch['attention_mask'][:,-width:].bool()
pad_values=prototype.batch['responses'][inactive].unique()
assert len(pad_values)==1
pad=int(pad_values[0])
report={'scope':__doc__,'model_GPU_calls':0,'carriers':[],
    'derivative_transport':'Original native backward coefficient is A/(16*N_carrier_B4). Diagnostic A_legacy=alpha*N_carrier_B4/N_historical_B4; A_common=alpha*16*N_carrier_B4/N_historical_global64. Thus exact positive-row derivatives use original target weights without pretending placeholder failure IDs are historical.',
    'limits':['Historical failure B4 counts are decoded approximations. Printed alpha is rounded; count inference independently calibrated. Iteration81 uses exact original mask counts and target alpha.',
        'Each checkpoint measures the entry old-policy target gradient of all four minibatches with frozen weights. It does not replay optimizer updates between those minibatches.',
        'Failure/source gradients are not measured by this carrier; no historical complete batch or actual training trajectory is fabricated.']}
for iteration in (6,41,81):
    saved=selected[iteration]
    assert not saved['success_truncated'],iteration
    original=moments[iteration-1]
    data=prototype.select(deepcopy=True)
    data.batch=prototype.batch.clone()
    target=torch.zeros_like(data.batch['response_mask'],dtype=torch.bool)
    positive_native_exact=[]
    for i,row in saved['positions'].items():
        length=row['prompt_length']
        assert length<=prompt_width
        original_ids=row['input_ids'].to(dtype=data.batch['input_ids'].dtype)
        assert len(original_ids)-length<=width
        prompt=original_ids[:length]
        response=original_ids[length:]
        data.batch['prompts'][i].fill_(pad)
        data.batch['prompts'][i,-length:]=prompt
        data.batch['responses'][i].fill_(pad)
        data.batch['responses'][i,:len(response)]=response
        data.batch['input_ids'][i]=torch.cat((data.batch['prompts'][i],data.batch['responses'][i]))
        data.batch['attention_mask'][i].zero_()
        data.batch['attention_mask'][i,prompt_width-length:prompt_width+len(response)]=1
        data.batch['response_mask'][i].zero_()
        data.batch['response_mask'][i,:len(response)]=row['policy_mask'][:len(response)].to(data.batch['response_mask'].dtype)
        data.batch['loss_mask'][i].zero_()
        data.batch['loss_mask'][i,prompt_width:]=data.batch['response_mask'][i]
        # The original owner assigns continuous response positions even to
        # right padding. All task initial prompts here equal the original
        # prototype; keep its actual native positions instead of recomputing
        # a different padding convention from the response attention mask.
        assert torch.equal(data.batch['prompts'][i],prototype.batch['prompts'][i])
        target[i,:len(response)]=row['target_mask'][:len(response)].bool()
        if iteration==81:
            fields=['prompts','responses','input_ids','attention_mask','response_mask','loss_mask','position_ids']
            equal={k:torch.equal(data.batch[k][i],prototype.batch[k][i]) for k in fields}
            positive_native_exact.append({'row':i,'fields_exact':equal})
            assert all(equal.values()),(i,equal)
    counts=list(saved['policy_counts_approx'])
    if iteration==81:
        counts=prototype.batch['response_mask'].sum(-1).tolist()
        alpha=float(prototype.batch['advantages'][target].max())
    else:
        alpha=original['native_whitened_maximum_rounded']
    legacy=torch.zeros_like(data.batch['advantages'])
    common=torch.zeros_like(legacy)
    carrier_counts=data.batch['response_mask'].sum(-1).tolist()
    B4=[]
    for start in range(0,256,4):
        mini=(start%128)//32
        indices=list(range(mini*32,mini*32+32))+list(range(128+mini*32,128+mini*32+32))
        n_h=sum(counts[start:start+4]);n_c=sum(carrier_counts[start:start+4]);n_g=sum(counts[i] for i in indices)
        legacy[start:start+4]=target[start:start+4]*alpha*n_c/n_h
        common[start:start+4]=target[start:start+4]*alpha*16*n_c/n_g
        B4.append({'start':start,'historical_B4_policy_count':n_h,'carrier_B4_policy_count':n_c,
            'historical_global64_policy_count':n_g,'selected_targets':int(target[start:start+4].sum())})
    assert not (target&~data.batch['response_mask'].bool()).any()
    data.batch['advantages']=legacy
    data.batch['diagnostic_target_legacy']=legacy
    data.batch['diagnostic_target_common']=common
    data.batch['diagnostic_target_mask']=target
    data.meta_info['multi_turn']=True
    data.meta_info['diagnostic_iteration']=iteration
    data.meta_info.pop('global_batch_info',None)
    path=DEST/f'iteration{iteration}-target-derivative-carrier.pkl'
    data.save_to_disk(str(path))
    report['carriers'].append({'iteration':iteration,'checkpoint':iteration-1,'path':str(path),
        'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'positive_rows':len(saved['positions']),
        'zero_derivative_placeholder_rows':256-len(saved['positions']),'native_target_alpha':alpha,
        'alpha_scope':'exact native' if iteration==81 else 'printed native maximum, rounded to0.001',
        'B4':B4,'native81_positive_fields_calibration':positive_native_exact})
assert not torch.cuda.is_initialized()
path=DEST/'carrier-receipt.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
