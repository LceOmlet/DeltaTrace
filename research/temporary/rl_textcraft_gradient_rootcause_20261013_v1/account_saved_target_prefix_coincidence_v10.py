"""Count repeated target score contexts in the existing85 positive rollout records.

This is input/coefficient accounting, not a gradient implementation or a new
training experiment. Nonmatching contexts' gradient inner products are unknown.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time

import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
account=json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())
inference=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())
report={'unix':time.time(),'scope':__doc__,'GPU_model_calls':0,'rows':[],
        'limits':[
            'Only positive saved target contexts are counted; this is not a complete total-gradient analysis.',
            'A repeated full ID prefix and next token identifies the same score function in exact arithmetic, not bitwise equality across native B4 numerical layouts.',
            'Squared coefficient sums are a descriptor of coinciding contexts; they are not parameter gradient norms, a norm lower bound, or an independent causal percentage.',
            'Historical B4 counts involving zero-reward rows remain approximate; native81 has an exact separate calibration.',
            'Distinct context gradient vectors need not be orthogonal. This audit cannot rule out alignment across different contexts.',
        ]}
for previous,white in zip(account['rows'],inference['rows']):
    step=previous['step']
    assert step==white['iteration'] if 'iteration' in white else step==white.get('step',step)
    saved={}
    for path in white['raw_files']:
        for row in torch.load(path,map_location='cpu',weights_only=False)['rows']:
            saved.setdefault(row['traj_uid'],row)
    contexts={}
    for uid,row in saved.items():
        prompt=row['prompt_length'];ids=row['input_ids'].tolist()
        target=row['target_mask'].bool()[:10240]
        selected=[];h=hashlib.sha256()
        for position,token in enumerate(ids[:prompt+len(target)]):
            if position>=prompt and bool(target[position-prompt]):
                selected.append((h.digest(),token))
            h.update(struct.pack('<I',token))
        contexts[uid]=selected
    permini=[]
    alpha=white['native_whitened_maximum_rounded']
    for mini in range(4):
        grouped=defaultdict(lambda:[0,0.0,0.0])
        within_B4=defaultdict(lambda:[0,0.0,0.0])
        for position in list(range(mini*32,(mini+1)*32))+list(range(128+mini*32,128+(mini+1)*32)):
            item=previous['positions'][position]
            if item['reward']==0:continue
            keys=contexts[item['representative_UID']]
            assert len(keys)==item['target_tokens'],(step,position)
            n=previous['B4'][position//4]['policy_count']
            coefficient=alpha/(16*n)
            for key in keys:
                for bucket in [grouped[key],within_B4[(position//4,key)]]:
                    bucket[0]+=1;bucket[1]+=coefficient;bucket[2]+=coefficient**2
        diagonal=sum(v[2] for v in grouped.values())
        if not diagonal:continue
        permini.append({'mini':mini,'positive_target_occurrences':sum(v[0] for v in grouped.values()),
            'distinct_full_ID_prefix_and_target_token':len(grouped),
            'occurrences_in_repeated_contexts':sum(v[0] for v in grouped.values() if v[0]>1),
            'maximum_context_multiplicity':max(v[0] for v in grouped.values()),
            'coefficient_diagonal_energy':diagonal,
            'coincident_context_coefficient_energy':sum(v[1]**2 for v in grouped.values()),
            'coincident_to_diagonal_coefficient_energy_ratio':sum(v[1]**2 for v in grouped.values())/diagonal,
            'same_B4_coincident_to_diagonal_coefficient_energy_ratio':sum(v[1]**2 for v in within_B4.values())/diagonal})
    report['rows'].append({'iteration':step,'native_grad_norm':previous['native_grad_norm'],
        'truncated_successes':previous['truncated_successes'],'minibatches':permini,
        'mean_context_to_diagonal_energy_ratio':statistics.mean(v['coincident_to_diagonal_coefficient_energy_ratio'] for v in permini),
        'mean_same_B4_context_to_diagonal_energy_ratio':statistics.mean(v['same_B4_coincident_to_diagonal_coefficient_energy_ratio'] for v in permini),
        'positive_target_repeated_context_fraction':sum(v['occurrences_in_repeated_contexts'] for v in permini)/sum(v['positive_target_occurrences'] for v in permini),
        'positive_target_native_context_sources':white['raw_files']})
report['windows']=[]
for lo,hi in [(1,10),(11,20),(31,40),(51,60),(61,70),(71,80),(81,85)]:
    rows=[r for r in report['rows'][lo-1:hi] if not r['truncated_successes']]
    report['windows'].append({'iterations':[lo,hi],'included_iterations':[r['iteration'] for r in rows],
        'medians':{key:statistics.median(r[key] for r in rows) for key in [
            'native_grad_norm','mean_context_to_diagonal_energy_ratio',
            'mean_same_B4_context_to_diagonal_energy_ratio','positive_target_repeated_context_fraction']}})
assert not torch.cuda.is_initialized()
report['finished_unix']=time.time()
dest=OUT/'saved-target-prefix-coincidence-v10.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
