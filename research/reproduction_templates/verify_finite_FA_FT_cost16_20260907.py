"""Audit fresh complete costs and permit only exact-vector historical metric reuse."""
import hashlib,json,statistics,math
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and p==json.loads((A/'finite_FA_FT_cost16_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'finite_FA_FT_cost16_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==digest==sha(A/name)
parents=[]
for name in ['required_parent','quality_parent']:
    path=A/'snapshot'/p[name].lstrip('/');assert sha(path)==p[name+'_sha256'];parents.append(json.loads(path.read_text()))
old,quality=parents
assert all(x['checkpoint_before']==x['checkpoint_after']==d['checkpoint_before']==d['checkpoint_after'] for x in parents)
eos_receipt=json.loads((F/'tokenizer_eos_receipt.json').read_text())
checkpoint_hashes={x['file']:x['sha256'] for x in d['checkpoint_before']}
assert all(checkpoint_hashes[n]==digest for n,digest in eos_receipt['files'].items())
assert eos_receipt['eos_token']=='<|im_end|>' and eos_receipt['eos_token_id']==151645
assert all(x['native_sources_before']==x['native_sources_after']==d['native_sources_before']==d['native_sources_after'] for x in parents)
assert d['dtype']=='torch.float16' and len(d['records'])==16
for key,value in p['budget'].items():assert d[key]==value,(key,d[key],value)
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),'scope':p['purpose'],
     'budget':p['budget'],'cases':[],'dataset_summary':{},'job_elapsed_seconds':d['elapsed_seconds'],
     'library_initialization_seconds':d['finite_library_initialization_seconds'],'new_quality_queries':0}
allcost=[];numeric=[];finite_calls=0;ft_calls=0
def public_cost(cost):return {k:v for k,v in cost.items() if k!='actual_root_input_ids'}
for case_number,(r,identity) in enumerate(zip(d['records'],p['selection'])):
    assert [r['dataset'],r['idx']]==identity and r['complete'] and len(r['runs'])==36
    prior=next(x for x in old['records'] if (x['dataset'],x['idx'])==tuple(identity))
    qp=next(x for x in quality['records'] if (x['dataset'],x['idx'])==tuple(identity))
    for k in ['input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions']:assert r[k]==prior[k]==qp[k]
    assert r['N']==len(prior['input_ids']) and hashlib.sha256(json.dumps(prior['input_ids']).encode()).hexdigest()==r['input_ids_sha256']
    for repeat in range(4):
        rows=r['runs'][repeat*9:(repeat+1)*9];offset=(case_number+repeat)%9
        assert [x['mode'] for x in rows]==p['methods'][offset:]+p['methods'][:offset]
        assert all(x['repeat']==repeat and x['warmup']==(repeat==0) for x in rows)
    profiles=[]
    for mode,v in r['profiles'].items():
        path=F/v['trace'];assert sha(path)==v['sha256']
        events=json.loads(path.read_text())['traceEvents'];names=[e['name'] for e in events if e.get('cat')=='kernel']
        counts={'native_FA_forward_kernels':sum('flash_fwd_kernel' in n for n in names),'finite_FA_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in names),'native_FA_backward_kernels':sum('flash_bwd' in n for n in names)}
        if mode=='finite':assert counts=={'native_FA_forward_kernels':108,'finite_FA_kernels':108,'native_FA_backward_kernels':0}
        else:assert counts['finite_FA_kernels']==0 and counts['native_FA_backward_kernels']==0
        profiles.append({'mode':mode,**counts})
        del events,names
    assert set(r['profiles'])==({'finite','both_1'} if case_number==0 else set())
    for mode,result in [(x['mode'],x['result']) for x in r['runs']]+[(m,x['result']) for m,x in r['profiles'].items()]:
        c=result['end_to_end_cost'];allcost.append(c)
        assert c['native_forwards']==1 and c['seconds']>0 and c['peak_allocated_bytes']>0 and c['vjps']==0
        assert len(result['score'])==len(r['user_positions']) and np.isfinite(result['score']).all()
        if mode=='finite':
            finite_calls+=1;baseline=list(prior['input_ids'])
            for j in r['eligible_positions']:baseline[j]=eos_receipt['eos_token_id']
            # Tokenizer EOS is proved by the parent baseline actual source and
            # original model tokenizer receipt; verify against the current call.
            assert c['actual_root_input_ids']==[[baseline,prior['input_ids']]]
            assert c['native_forward_trajectories']==2 and c['native_decoder_layer_calls']==72 and c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
            assert c['public_FA_activity']['auxiliary_completed']==36
            assert sum(x['calls_enqueued'] for x in c['finite_FA_activity'])==36
            assert result['per_operator_ledger_collected'] is False and result['ledger'] is None
            x=np.asarray(result['signed_full_sequence']);projected=np.maximum(x[r['user_positions']],0).astype(np.float32)
            assert np.array_equal(projected,np.asarray(result['score']))
        else:
            ft_calls+=1
            assert c['actual_root_input_ids']==[[prior['input_ids']]] and c['native_forward_trajectories']==1 and c['native_decoder_layer_calls']==36 and c['extra_replay_calls']==0
    row={k:r[k] for k in ['dataset','idx','input_ids_sha256','N','prompt_len']};row.update(methods={},profiles=profiles)
    detail=dict(row,runs=[])
    for mode in p['methods']:
        runs=[x for x in r['runs'] if x['mode']==mode];measured=runs[1:]
        # Rotated ordering leaves the per-method repeat ordering ascending.
        assert [x['repeat'] for x in runs]==[0,1,2,3]
        costs=[x['result']['end_to_end_cost'] for x in measured]
        scores=[x['result']['score'] for x in runs]
        if mode=='finite':old_score=qp['scores']['finite'];metrics=qp['metrics']['finite'];src=p['quality_parent_sha256']
        else:
            family,hop=mode.rsplit('_',1);key=f'flashtrace_{family}_hop{hop}';old_score=prior['scores'][key];metrics=prior['metrics'][key];src=p['required_parent_sha256']
        exact=scores[1]==old_score
        reuse=r['historical_quality_reuse'][mode]
        assert reuse=={'exact_projected_score_match':exact,'source_sha256':src,'metrics':metrics if exact else None,'new_quality_queries':0}
        row['methods'][mode]={'median_seconds':statistics.median(c['seconds'] for c in costs),'all_measured_seconds':[c['seconds'] for c in costs],
            'peak_bytes':max(c['peak_allocated_bytes'] for c in costs),'warmup_seconds':runs[0]['result']['end_to_end_cost']['seconds'],
            'repeat_scores_identical':all(x==scores[1] for x in scores),'projected_score_exactly_matches_quality_parent':exact,
            'quality_metrics':metrics if exact else None,'quality_source_sha256':src,
            'relative_score_L2_to_parent':float(np.linalg.norm(np.array(scores[1])-old_score)/max(np.linalg.norm(old_score),1e-30))}
        for x in runs:
            item={'mode':mode,'repeat':x['repeat'],'warmup':x['warmup'],'score':x['result']['score'],'cost':public_cost(x['result']['end_to_end_cost'])}
            if mode=='finite':item['signed_full_sequence']=x['result']['signed_full_sequence'];item['endpoint_scores32']=x['result']['endpoint_scores32'];item['unassigned_total']=x['result']['unassigned_total']
            detail['runs'].append(item)
    f=row['methods']['finite'];ft=row['methods']['both_1']
    row['finite_to_both1_median_ratio']=f['median_seconds']/ft['median_seconds']
    row['finite_not_slower_than_both1']=f['median_seconds']<=ft['median_seconds']
    row['FT_controls_within_candidate_latency']=[m for m in p['methods'] if m!='finite' and row['methods'][m]['median_seconds']<=f['median_seconds']]
    row['budget_matched_quality_oracle']={}
    for metric,larger in [('recovery',True),('rise',False),('mas',False)]:
        eligible=[m for m in row['FT_controls_within_candidate_latency'] if row['methods'][m]['quality_metrics'] is not None and row['methods'][m]['quality_metrics'][metric] is not None]
        if eligible and f['quality_metrics'] is not None and f['quality_metrics'][metric] is not None:
            best=(max if larger else min)(eligible,key=lambda m:row['methods'][m]['quality_metrics'][metric])
            row['budget_matched_quality_oracle'][metric]={'control':best,'control_metric':row['methods'][best]['quality_metrics'][metric],
                'candidate_metric':f['quality_metrics'][metric],'candidate_minus_oracle':f['quality_metrics'][metric]-row['methods'][best]['quality_metrics'][metric]}
    out['cases'].append(row);numeric.append(detail)
assert finite_calls==65 and ft_calls==513 and len(allcost)==578
for ds in ['niah_mq_q2','morehopqa']:
    rows=[r for r in out['cases'] if r['dataset']==ds];assert len(rows)==8
    group={'methods':{},'finite_to_both1_paired_median_ratio':statistics.median(r['finite_to_both1_median_ratio'] for r in rows),
        'finite_slower_than_both1_indices':[r['idx'] for r in rows if not r['finite_not_slower_than_both1']],
        'finite_to_both1_total_median_time_ratio':sum(r['methods']['finite']['median_seconds'] for r in rows)/sum(r['methods']['both_1']['median_seconds'] for r in rows),
        'budget_matched_oracle_deltas':{}}
    for mode in p['methods']:
        group['methods'][mode]={'mean_seconds':statistics.mean(r['methods'][mode]['median_seconds'] for r in rows),
            'maximum_peak_bytes':max(r['methods'][mode]['peak_bytes'] for r in rows),'quality':{}}
        for metric in ['rise','mas','recovery']:
            values=[r['methods'][mode]['quality_metrics'][metric] for r in rows if r['methods'][mode]['quality_metrics'] is not None and r['methods'][mode]['quality_metrics'][metric] is not None]
            group['methods'][mode]['quality'][metric]={'count':len(values),'mean':statistics.mean(values) if len(values)==8 else None}
    for metric in ['rise','mas','recovery']:
        deltas=[r['budget_matched_quality_oracle'][metric]['candidate_minus_oracle'] for r in rows if metric in r['budget_matched_quality_oracle']]
        group['budget_matched_oracle_deltas'][metric]={'count':len(deltas),'mean':statistics.mean(deltas) if len(deltas)==8 else None,'paired_deltas':deltas}
    out['dataset_summary'][ds]=group
out['finite_to_both1_paired_median_ratio']=statistics.median(r['finite_to_both1_median_ratio'] for r in out['cases'])
out['finite_slower_than_both1_cases']=[[r['dataset'],r['idx']] for r in out['cases'] if not r['finite_not_slower_than_both1']]
out['sum_all_measured_scope_timer_seconds']=sum(c['seconds'] for c in allcost)
out['limits']='Original16 development B1 latency comparison only; candidate root B2 holds two endpoints for one example. FT is unchanged author eager implementation; candidate uses nativeFA plus declared finite propagation. Historical quality links require exact current repeat1 projected score equality, not new curves or independence. Budget oracle chooses different FT variants per case/metric after scoring and is a diagnostic upper envelope, not an implementable method. Cold warmups/profile/I-O/checkpoint costs are separate; all calls counted. No sign repair or new sign claim.'
(A/'finite_FA_FT_cost16_summary_20260907.json').write_text(json.dumps(out,indent=2))
(A/'finite_FA_FT_cost16_numeric_20260907.json').write_text(json.dumps({'raw_sha256':out['raw_sha256'],'records':numeric},separators=(',',':')))
print(json.dumps({k:v for k,v in out.items() if k not in ['cases','dataset_summary']},indent=2))
print(json.dumps(out['dataset_summary'],indent=2))
