"""Verify seq-only original-function fidelity and fair same-job full costs."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_original_FT_seq_cost16_20260907_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and p==json.loads((A/'original_FT_seq_cost16_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'original_FT_seq_cost16_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==digest==sha(A/name)
parents=[]
for key in ['required_parent','quality_parent']:
    file=A/'snapshot'/p[key].lstrip('/');assert sha(file)==p[key+'_sha256'];parents.append(json.loads(file.read_text()))
old,quality=parents
assert all(x['checkpoint_before']==x['checkpoint_after']==d['checkpoint_before']==d['checkpoint_after'] for x in parents)
assert all(x['native_sources_before']==x['native_sources_after']==d['native_sources_before']==d['native_sources_after'] for x in parents)
for k,v in p['budget'].items():assert d[k]==v,(k,d[k],v)
assert d['dtype']=='torch.float16' and len(d['records'])==16
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),'scope':p['purpose'],'budget':p['budget'],'cases':[],'job_elapsed_seconds':d['elapsed_seconds']}
numeric=[];allcost=[];counts={m:0 for m in p['methods']}
for ci,(r,selection) in enumerate(zip(d['records'],p['selection'])):
    assert [r['dataset'],r['idx']]==selection and r['complete'] and len(r['runs'])==12
    prior=next(x for x in old['records'] if (x['dataset'],x['idx'])==tuple(selection));qp=next(x for x in quality['records'] if (x['dataset'],x['idx'])==tuple(selection))
    for k in ['input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions']:assert r[k]==prior[k]==qp[k]
    assert hashlib.sha256(json.dumps(prior['input_ids']).encode()).hexdigest()==r['input_ids_sha256']
    for repeat in range(4):
        offset=(ci+repeat)%3;group=r['runs'][3*repeat:3*(repeat+1)]
        assert [x['mode'] for x in group]==p['methods'][offset:]+p['methods'][:offset]
        assert all(x['repeat']==repeat and x['warmup']==(repeat==0) for x in group)
    profile={}
    for mode,v in r['profiles'].items():
        c=v['original_python_calls'];assert c['normalize_sum_to_one']>=1
        if mode=='seq_1':assert c.get('compute_CAGE_token_attr',0)==c.get('get_all_token_attrs',0)==0
        else:assert c['get_all_token_attrs']==1 and c['compute_CAGE_token_attr']>0
        profile[mode]=c
    assert set(profile)==({'seq_1','both_1'} if ci==0 else set())
    detail={'dataset':r['dataset'],'idx':r['idx'],'input_ids_sha256':r['input_ids_sha256'],'runs':[],'profiles':profile}
    for mode,v in [(x['mode'],x['result']) for x in r['runs']]+[(m,x['result']) for m,x in r['profiles'].items()]:
        c=v['end_to_end_cost'];allcost.append(c);counts[mode]+=1
        assert c['native_forwards']==1 and c['vjps']==0 and c['seconds']>0 and np.isfinite(v['score']).all()
        if mode=='finite':
            baseline=list(prior['input_ids'])
            for j in r['eligible_positions']:baseline[j]=151645
            assert c['actual_root_input_ids']==[[baseline,prior['input_ids']]] and c['native_forward_trajectories']==2
            assert c['native_decoder_layer_calls']==72 and c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
            assert c['public_FA_activity']['auxiliary_completed']==36 and sum(x['calls_enqueued'] for x in c['finite_FA_activity'])==36
            assert np.array_equal(np.maximum(np.array(v['signed_full_sequence'])[r['user_positions']],0).astype(np.float32),np.asarray(v['score']))
        else:
            assert c['actual_root_input_ids']==[[prior['input_ids']]] and c['native_forward_trajectories']==1
            assert c['native_decoder_layer_calls']==36 and c['extra_replay_calls']==0
    row={k:r[k] for k in ['dataset','idx','input_ids_sha256','N','prompt_len']};row.update(methods={},original_python_profiles=profile)
    for mode in p['methods']:
        runs=[x for x in r['runs'] if x['mode']==mode];assert [x['repeat'] for x in runs]==[0,1,2,3]
        score=runs[1]['result']['score'];ref=qp['scores']['finite'] if mode=='finite' else prior['scores']['flashtrace_both_hop1']
        metric=qp['metrics']['finite'] if mode=='finite' else prior['metrics']['flashtrace_both_hop1'];exact=score==ref
        assert r['historical_quality_reuse'][mode]=={'exact_projected_score_match':exact,'metrics':metric if exact else None,'new_quality_queries':0}
        costs=[x['result']['end_to_end_cost'] for x in runs[1:]]
        row['methods'][mode]={'median_seconds':statistics.median(c['seconds'] for c in costs),'measured_seconds':[c['seconds'] for c in costs],'peak_bytes':max(c['peak_allocated_bytes'] for c in costs),'warmup_seconds':runs[0]['result']['end_to_end_cost']['seconds'],'all_repeat_scores_identical':all(x['result']['score']==score for x in runs),'exact_projected_score_match':exact,'quality_metrics':metric if exact else None}
        for x in runs:
            value=x['result'];item={k:x[k] for k in ['mode','repeat','warmup']};item.update(score=value['score'],cost={k:v for k,v in value['end_to_end_cost'].items() if k!='actual_root_input_ids'})
            if mode=='finite':item.update(signed_full_sequence=value['signed_full_sequence'],endpoint_scores32=value['endpoint_scores32'],unassigned_total=value['unassigned_total'])
            detail['runs'].append(item)
    seq=[x['result']['score'] for x in r['runs'] if x['mode']=='seq_1'];full=[x['result']['score'] for x in r['runs'] if x['mode']=='both_1']
    row['all_seq_vectors_exactly_match_full_runner']=all(a==b for a,b in zip(seq,full))
    v=row['methods'];row['finite_to_full_runner_ratio']=v['finite']['median_seconds']/v['both_1']['median_seconds'];row['finite_to_seq_only_ratio']=v['finite']['median_seconds']/v['seq_1']['median_seconds'];row['seq_to_full_runner_ratio']=v['seq_1']['median_seconds']/v['both_1']['median_seconds']
    out['cases'].append(row);numeric.append(detail)
assert counts=={'finite':64,'both_1':65,'seq_1':65} and len(allcost)==194
for label in ['finite_to_full_runner_ratio','finite_to_seq_only_ratio','seq_to_full_runner_ratio']:out['paired_median_'+label]=statistics.median(r[label] for r in out['cases'])
out['seq_vector_matches_full_runner_cases']=sum(r['all_seq_vectors_exactly_match_full_runner'] for r in out['cases'])
out['finite_slower_than_seq_only_cases']=[[r['dataset'],r['idx'],r['finite_to_seq_only_ratio']] for r in out['cases'] if r['finite_to_seq_only_ratio']>1]
out['sum_attribution_timer_seconds']=sum(c['seconds'] for c in allcost)
out['limits']='Original16 development B1 only, with current model/FA precision. Direct seq-only is composition of unchanged original class and normalization, skips unused row/rec views without changing attention or score rules; exact-vector reuse conditions are explicit. No new quality curves, independent confirmation, B4 speed claim or new negative-sign assertion. Original full-runner costs remain valid only for that interface.'
(A/'original_FT_seq_cost16_summary_20260907.json').write_text(json.dumps(out,indent=2));(A/'original_FT_seq_cost16_numeric_20260907.json').write_text(json.dumps({'raw_sha256':out['raw_sha256'],'records':numeric},separators=(',',':')))
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
