"""Recompute actual conditional effects and report signed-allocation limitations."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_finite_P1_conditional_sign_review_20260907_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and p==json.loads((A/'finite_P1_conditional_sign_review_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'finite_P1_conditional_sign_review_20260907.py')
parent_path=A/'snapshot'/p['parent'].lstrip('/');assert sha(parent_path)==p['parent_sha256'];parent=json.loads(parent_path.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
for k,v in p['budget'].items():assert d[k]==v
assert d['dtype']=='torch.float16' and len(d['records'])==16
rng=np.random.default_rng(730907);observed=[];forward_calls=[];case_details=[]
for row,selection in zip(d['records'],p['selection']):
    assert (row['dataset'],row['idx'],row['input_ids_sha256'])==(selection['dataset'],selection['idx'],selection['input_ids_sha256'])
    original=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    assert len(original['input_ids'])==row['N'] and original['prompt_len']==row['prompt_len']
    finite=next(r['result'] for r in original['runs'] if r['mode']=='finite' and r['repeat']==1)
    x=np.asarray(finite['signed_full_sequence']);eligible=original['eligible_positions']
    pos=sorted((j for j in eligible if x[j]>0),key=lambda j:(-x[j],j))[:8]
    neg=sorted((j for j in eligible if x[j]<0),key=lambda j:(x[j],j))[:8]
    rest=sorted(set(eligible)-set(pos)-set(neg));random=rng.choice(rest,8,replace=False).tolist()
    positions=pos+neg+random
    assert positions==[r['position'] for r in row['selected']] and row['selected']==selection['selected']
    assert all(r['finite_P1_score']==x[r['position']] for r in row['selected'])
    effects={};drift={}
    for background in ['clean','eos']:
        repeats=row['baseline_repeats'][background];assert len(repeats)==2
        forward_calls.extend(repeats)
        before=np.asarray(repeats[0]['scores32_sum64']);after=np.asarray(repeats[1]['scores32_sum64'])
        drift[background]={'maximum_absolute_repeat_score_change':float(abs(after-before).max()),'first_baseline_row_spread':float(before.max()-before.min()),
            'token_values_repeat_exact':repeats[0]['target_logprobs32']==repeats[1]['target_logprobs32']}
        queries=[r for r in row['interventions'] if r['background']==background];assert len(queries)==6
        assert sum([r['positions'] for r in queries],[])==positions
        for query in queries:
            assert query['lanes']==[0,1,2,3] and query['actual_changed_positions_checked'] and query['original_target_preserved']
            forward_calls.append(query['result']);q=np.asarray(query['result']['scores32_sum64'])
            true_effect=before-q if background=='clean' else q-before
            repeated_effect=after-q if background=='clean' else q-after
            assert np.array_equal(true_effect,np.asarray(query['conditional_effects']))
            for lane,j in enumerate(query['positions']):effects[(j,background)]={'effect':float(true_effect[lane]),'effect_if_end_baseline_used':float(repeated_effect[lane]),'lane':lane}
    for selection in row['selected']:
        item=dict(selection,dataset=row['dataset'],idx=row['idx'])
        item.update({background:effects[(selection['position'],background)] for background in ['clean','eos']})
        observed.append(item)
    detail={'dataset':row['dataset'],'idx':row['idx'],'baseline_checks':drift,
        'conditional_sign_reversals':sum(effects[(j,'clean')]['effect']*effects[(j,'eos')]['effect']<0 for j in positions)}
    if 'profile' in row:
        profile=row['profile'];forward_calls.append(profile['result']);path=F/profile['trace'];assert sha(path)==profile['sha256']
        events=json.loads(path.read_text())['traceEvents'];names=[e['name'] for e in events if e.get('cat')=='kernel']
        assert sum('flash_fwd_kernel' in n for n in names)==36 and not any('flash_bwd' in n or 'deltatrace_fa_finite' in n for n in names)
        detail['profile']={'actual_default_FA_forward_kernels':36,'native_backward_or_finite_kernels':0};del events,names
    case_details.append(detail)
assert len(forward_calls)==257 and len(observed)==384
for call in forward_calls:
    values=np.asarray(call['target_logprobs32']);assert values.ndim==2 and values.shape[0]==4 and np.isfinite(values).all()
    assert all(abs(math.fsum(lp)-g)<1e-9 for lp,g in zip(values,call['scores32_sum64']))
    cost=call['cost'];assert cost['native_forwards']==1 and cost['native_forward_trajectories']==4 and cost['native_decoder_layer_calls']==36 and cost['extra_replay_calls']==0 and cost['vjps']==0
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),'scope':p['purpose'],'budget':p['budget'],
    'selected_tokens':384,'conditional_effects':768,'cases':case_details,'groups':{},'elapsed_job_seconds':d['elapsed_seconds'],
    'sum_native_forward_and_score_timer_seconds':sum(c['cost']['seconds'] for c in forward_calls),
    'maximum_recorded_peak_bytes':max(c['cost']['peak_allocated_bytes'] for c in forward_calls)}
for dataset in ['niah_mq_q2','morehopqa']:
    out['groups'][dataset]={}
    for kind in ['positive','negative','random']:
        rows=[r for r in observed if r['dataset']==dataset and r['selection']==kind];assert len(rows)==64
        item={'tokens':64,'allocation_positive':sum(r['finite_P1_score']>0 for r in rows),'allocation_negative':sum(r['finite_P1_score']<0 for r in rows),'allocation_zero':sum(r['finite_P1_score']==0 for r in rows),
            'sign_changes_between_conditions':sum(r['clean']['effect']*r['eos']['effect']<0 for r in rows)}
        for background in ['clean','eos']:
            effect=np.array([r[background]['effect'] for r in rows]);allocation=np.array([r['finite_P1_score'] for r in rows]);sg=np.sign(effect);pred=np.sign(allocation)
            item[background]={'same_sign_count':int((sg==pred).sum()),'opposite_nonzero_sign_count':int((sg*pred<0).sum()),'zero_effect_count':int((sg==0).sum()),
                'positive_effect_count':int((sg>0).sum()),'negative_effect_count':int((sg<0).sum()),'median_effect':float(np.median(effect)),
                'mean_absolute_effect':float(np.mean(abs(effect))),
                'fraction_absolute_effect_mass_on_matching_signs':float(abs(effect[sg==pred]).sum()/max(abs(effect).sum(),1e-30)),
                'effects_that_reverse_if_end_baseline_used':int(sum(np.sign(r[background]['effect'])!=np.sign(r[background]['effect_if_end_baseline_used']) for r in rows))}
        out['groups'][dataset][kind]=item
out['limits']='Original16 development, deliberately selected64tokens per dataset/category. Conditional effects are direct finite native values at explicit backgrounds and matched B4 lanes, not additive/universal token credit. No sign confidence interval, new quality benchmark, independent confirmation or candidate retuning. Timer excludes result serialization; full job elapsed separately reports it. Baseline repeats characterize observed drift only, not universal numerical bounds.'
(A/'finite_P1_conditional_sign_summary_20260907.json').write_text(json.dumps(out,indent=2))
(A/'finite_P1_conditional_sign_numeric_20260907.json').write_text(json.dumps({'raw_sha256':out['raw_sha256'],'observed_token_effects':observed,'records':d['records']},separators=(',',':')))
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
