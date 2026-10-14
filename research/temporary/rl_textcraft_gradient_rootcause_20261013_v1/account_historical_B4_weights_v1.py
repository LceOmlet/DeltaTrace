"""Recover original dump order and quantify B4 weights from existing artifacts.

Positive IDs/masks come from original passive records. Decoded failure counts
are an offline approximation, calibrated on all original iteration81 rows and
against independent historical whitening-count intervals. No actor inputs,
credit, training, optimizer, rollout, or model calls are changed or executed.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],trust_remote_code=True,local_files_only=True)
dump_dir=Path(launch['options']['trainer.rollout_data_dir'])
inference=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())
native=json.loads((OUT/'temporal-native-reports-v1.json').read_bytes())
traces=[]
for step in range(85):
    rows={}
    for log in sorted(native['raw_reports']):
        for tr in native['raw_reports'][log][step]['value']['traces']:
            rows.setdefault(tr['traj_uid'],tr)
    traces.append(rows)
pattern=re.compile(r'\n(user|assistant)\n')
suffix_n=len(tok.encode('<|im_end|>',add_special_tokens=False))

def decoded_count(text):
    parts=pattern.split(text)
    n=len(tok.encode(parts[0],add_special_tokens=False))+suffix_n if parts[0] else 0
    turns=[]
    for role,content in zip(parts[1::2],parts[2::2]):
        if role=='assistant':
            k=len(tok.encode(content,add_special_tokens=False))+suffix_n
            turns.append(k);n+=k
    return n,turns

result={'started_unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'rows':[],
    'limits':['Decoded failure counts can differ because special tokens disappear, a cap can cut the last assistant suffix, and literal role strings can occur inside generated content.',
        'At iteration81 all256 native masks provide a complete calibration; earlier totals are compared with independently inferred whitening-count intervals, not declared exact.',
        'Successful-row raw IDs/masks and target NLLs are native saved evidence; ambiguous decoded duplicates are checked for equivalent IDs/masks but UID itself is not inferred as unique.',
        'Output-logit bounds use recorded factual DT endpoint probabilities, not parameter gradients. Their temporal shape is an accounting diagnostic, not a complete causal explanation.']}
selected={}
for step in range(1,86):
    original=inference['rows'][step-1]
    dump_path=dump_dir/f'{step}.jsonl'
    dumps=[json.loads(s) for s in dump_path.read_text().splitlines()]
    assert len(dumps)==256
    saved={}
    for file in original['raw_files']:
        for row in torch.load(file,map_location='cpu',weights_only=False)['rows']:
            saved.setdefault(row['traj_uid'],row)
    by_text=defaultdict(list)
    for uid,row in saved.items():
        response=row['input_ids'][row['prompt_length']:][:10240]
        text=tok.decode(response,skip_special_tokens=True)
        by_text[text].append((uid,row))
    counts=[];positions=[];positive=[];unmatched=[];ambiguous=[]
    for i,dump in enumerate(dumps):
        approx,turns=decoded_count(dump['output'])
        count=approx
        row_record={'position':i,'reward':dump['score'],'decoded_policy_count':approx,
            'assistant_count':len(turns),'max_decoded_turn_count':max(turns,default=0)}
        if dump['score']!=0:
            candidates=by_text.get(dump['output'],[])
            if not candidates:
                unmatched.append(i)
            else:
                uid,row=candidates[0]
                p=row['policy_mask'].bool()[:10240]
                t=row['target_mask'].bool()[:10240]
                count=int(p.sum())
                if len(candidates)>1:
                    identical=all(torch.equal(c[1]['input_ids'],row['input_ids']) and
                        torch.equal(c[1]['policy_mask'],row['policy_mask']) and
                        torch.equal(c[1]['target_mask'],row['target_mask']) for c in candidates)
                    ambiguous.append({'position':i,'candidate_UID_count':len(candidates),'IDs_masks_equivalent':identical})
                    assert identical,(step,i)
                tr=traces[step-1][uid]
                row_record.update(native_success_policy_count=count,target_tokens=int(t.sum()),
                    representative_UID=uid,candidate_UIDs=[c[0] for c in candidates],
                    factual_full_target_NLL=-tr['factual_target_logp'],
                    success_count_error=approx-count,
                    full_response_count=len(row['input_ids'])-row['prompt_length'])
                positive.append((i,row,tr))
        counts.append(count);positions.append(row_record)
    assert not unmatched,(step,unmatched)
    assert sum(d['score']!=0 for d in dumps)==original['success_count'],step
    native_calibration=None
    if step==81:
        ref=json.loads((OUT/'dump-policy-count-calibration-v1.json').read_bytes())['native81_calibration']['rows']
        native_counts=[r['native_policy_count'] for r in ref]
        native_calibration={'actual_counts':native_counts,'per_B4_approx_minus_native':[
            sum(counts[j:j+4])-sum(native_counts[j:j+4]) for j in range(0,256,4)],
            'N_approx':sum(counts),'N_native':sum(native_counts)}
    micros=[];minis=[]
    for start in range(0,256,4):
        rows=positions[start:start+4]
        n=sum(counts[start:start+4])
        assert n>0,(step,start)
        good=[r for r in rows if r['reward']!=0]
        micros.append({'start':start,'policy_count':n,'success_rows':len(good),
            'success_target_tokens':sum(r['target_tokens'] for r in good),
            'target_NLL_sum':sum(r['factual_full_target_NLL'] for r in good),
            'target_score_lower_sum':sum(-math.expm1(-r['factual_full_target_NLL']) for r in good),
            'suspect_large_decoded_turn':any(r['max_decoded_turn_count']>513 for r in rows)})
    alpha=original['native_whitened_maximum_rounded']
    for mini in range(4):
        m=micros[mini*8:mini*8+8]+micros[32+mini*8:32+mini*8+8]
        n=sum(r['policy_count'] for r in m)
        low=sum(r['target_score_lower_sum']/r['policy_count'] for r in m)
        high=sum(r['target_NLL_sum']/r['policy_count'] for r in m)
        common_low=sum(r['target_score_lower_sum'] for r in m)/n
        common_high=sum(r['target_NLL_sum'] for r in m)/n
        minis.append({'index':mini,'global_policy_count_approx':n,
            'B4_count_range':[min(r['policy_count'] for r in m),max(r['policy_count'] for r in m)],
            'target_coefficient_mass_legacy':alpha*sum(r['success_target_tokens']/r['policy_count'] for r in m)/16,
            'target_coefficient_mass_common':alpha*sum(r['success_target_tokens'] for r in m)/n,
            'target_logit_L1_bounds_legacy_DT_endpoint':[2*alpha*low/16,2*alpha*high/16],
            'target_logit_L1_bounds_common_DT_endpoint':[2*alpha*common_low,2*alpha*common_high],
            'successful_target_weight_ratio_max':max((n/(16*r['policy_count']) for r in m if r['success_target_tokens']),default=None)})
    output={'step':step,'dump_path':str(dump_path),'dump_SHA256':hashlib.sha256(dump_path.read_bytes()).hexdigest(),
        'native_grad_norm':original['native_grad_norm'],'truncated_successes':original['truncated_successes'],
        'positive_rows':len(positive),'positive_raw_files':original['raw_files'],
        'total_policy_count_approx':sum(counts),'positions':positions,'B4':micros,'global64':minis,
        'ambiguous_equivalent_decodes':ambiguous,'native81_count_calibration':native_calibration}
    result['rows'].append(output)
    if step in (6,41,81):
        selected[step]={'positions':{i:row for i,row,tr in positive},'policy_counts_approx':counts,
            'source_dump':str(dump_path),'success_truncated':original['truncated_successes']}
    if step%10==0:
        (OUT/'historical-B4-accounting-progress-v1.json').write_text(json.dumps({'completed_iterations':step,'unix':time.time(),'model_calls':0})+'\n')
assert not torch.cuda.is_initialized()
selected_path=OUT/'historical-success-ID-placement-v1.pt'
torch.save(selected,selected_path)
result['selected_exact_positive_artifact']={'path':str(selected_path),'SHA256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
    'scope':'Original native positive IDs/masks, matched to original dump positions at6/41/81; failure counts approximate. No actor/credit input has been created.'}
result['finished_unix']=time.time()
path=OUT/'historical-B4-weight-accounting-v1.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':result}))
