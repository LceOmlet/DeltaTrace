"""Transport saved positive target derivatives for16 matched initial tasks.

Native DataProto/HF/PPO own model and gradient computation. This only packs
already saved IDs/masks and preserves their recorded dL/dlogp coefficients.
No new credit values, rewards, loss implementation, or training data are made.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'matched-task-target-derivatives-v11'
assert not (DEST/'case-manifest.json').exists(),'Preserve prior prepared cases.'
inventory=json.loads((OUT/'matched-native-task-inventory-v11.json').read_bytes())
assert len(inventory['selected_tasks'])==16
src=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
prototype=DataProto.concat([DataProto.load_from_disk(s['input']) for s in src]).select_idxs(torch.arange(64))
width=prototype.batch['responses'].shape[1];prompt_width=prototype.batch['prompts'].shape[1]
pad_values=prototype.batch['responses'][~prototype.batch['attention_mask'][:,-width:].bool()].unique()
assert len(pad_values)==1;pad=int(pad_values[0])
cache={};cases=[];DEST.mkdir(exist_ok=True)
for name in ['early','late']:
    selected=[row for task in inventory['selected_tasks'] for row in task[name]]
    data=prototype.select(deepcopy=True);data.batch=prototype.batch.clone()
    fields={'prompts','responses','input_ids','attention_mask','position_ids','response_mask',
            'loss_mask','old_log_probs','ref_log_prob','advantages'}
    for key in list(data.batch.keys()):
        if key not in fields:data.batch.pop(key)
    target=torch.zeros_like(data.batch['response_mask'],dtype=torch.bool)
    for i,item in enumerate(selected):
        source=item['raw_source'];path=source['path']
        if path not in cache:cache[path]=torch.load(path,map_location='cpu',weights_only=False)['rows']
        row=cache[path][source['row_index']]
        assert row['traj_uid']==item['representative_UID']
        length=row['prompt_length'];ids=row['input_ids'].to(data.batch['input_ids'].dtype)
        response=ids[length:];assert len(response)<=width
        first=int(row['policy_mask'].bool().nonzero()[0])
        assert hashlib.sha256(ids[:length+first].numpy().tobytes()).hexdigest()==item['initial_prefix_ID_SHA256']
        data.batch['prompts'][i].fill_(pad);data.batch['prompts'][i,-length:]=ids[:length]
        assert torch.equal(data.batch['prompts'][i],prototype.batch['prompts'][i])
        data.batch['responses'][i].fill_(pad);data.batch['responses'][i,:len(response)]=response
        data.batch['input_ids'][i]=torch.cat([data.batch['prompts'][i],data.batch['responses'][i]])
        data.batch['attention_mask'][i].zero_();data.batch['attention_mask'][i,prompt_width-length:prompt_width+len(response)]=1
        data.batch['response_mask'][i].zero_();data.batch['response_mask'][i,:len(response)]=row['policy_mask'][:len(response)].to(data.batch['response_mask'].dtype)
        data.batch['loss_mask'][i].zero_();data.batch['loss_mask'][i,prompt_width:]=data.batch['response_mask'][i]
        target[i,:len(response)]=row['target_mask'][:len(response)].bool()
        assert int(target[i].sum())==item['target_tokens']
        # The original saved DT endpoint ends at its last causal target;
        # later zero-target policy slots are absent from this carrier. Its
        # original full-row denominator is retained in the transport below.
        assert int(data.batch['response_mask'][i].sum())<=item['policy_tokens']
    transported=torch.zeros_like(data.batch['advantages'])
    coefficients=[]
    for i,item in enumerate(selected):
        start=(i//4)*4;n_carrier=int(data.batch['response_mask'][start:start+4].sum())
        historical=item['original_target_alpha_rounded']/(16*item['original_B4_policy_count'])
        transported[i]=target[i]*item['original_target_alpha_rounded']*n_carrier/item['original_B4_policy_count']
        coefficients.append({'row':i,'historical_target_dL_dlogp_magnitude':historical,
                             'carrier_B4_policy_count':n_carrier,'original_source':item})
    assert not (target&~data.batch['response_mask'].bool()).any()
    data.batch['advantages']=transported;data.batch['diagnostic_target_legacy']=transported
    data.batch['diagnostic_target_mask']=target
    data.non_tensor_batch={'uid':np.asarray([r['initial_prefix_ID_SHA256'] for r in selected],dtype=object),
                          'traj_uid':np.asarray([r['representative_UID'] for r in selected],dtype=object)}
    data.meta_info['multi_turn']=True;data.meta_info.pop('global_batch_info',None)
    path=DEST/f'{name}-target-derivative-carrier.pkl';data.save_to_disk(str(path))
    cases.append({'name':name,'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'rows':64,'matched_task_states':16,'positive_reward_rows':64,'coefficients':coefficients})
assert [r['original_source']['initial_prefix_ID_SHA256'] for r in cases[0]['coefficients']]==[r['original_source']['initial_prefix_ID_SHA256'] for r in cases[1]['coefficients']]
report={'unix':time.time(),'destination':str(DEST),'scope':__doc__,'cases':cases,
        'question':inventory['question'],'why_needed':inventory['why_needed'],
        'GPU_model_calls':0,'copied_training_function_bodies':0,'checkpoint_for_both':5,
        'derivative_transport':'Native complete global64/B4 PG at its recomputed old-policy point uses A_carrier/(16*N_carrier). A_carrier=alpha_saved*N_carrier/N_saved_B4 preserves each historical target derivative coefficient; non-target derivatives are zero. This is a diagnostic carrier, never a proposed training/credit method.',
        'selection':inventory['selection'],
        'limits':['Both cases contain only original successful target derivatives; this does not recreate either whole training batch.',
                  'Early and late historical B4 denominator and alpha uncertainty is retained.',
                  'Matching initial tasks removes task-composition confounding but leaves actual generated histories and saved coefficient changes together.',
                  'The saved DT endpoint omits policy slots after the last target. Its causal IDs/masks are transported unchanged and its original full-row target derivative weights are retained.',
                  'No result constitutes fresh-training repair acceptance, or implies the outcome of training from the start with a different denominator.']}
assert not torch.cuda.is_initialized()
dest=DEST/'case-manifest.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
