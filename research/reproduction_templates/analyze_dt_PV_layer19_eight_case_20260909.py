"""CPU aggregation of four independently audited, fixed PV19 jobs.

No scoring/attribution or selection. All eight used development cases included.
Historical FT numbers remain explicitly associated with their original process.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import close,needle

A=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts(delta,lower=True):
    x=np.asarray(delta,dtype=np.float64)*(1 if lower else -1)
    return {'improved':int((x<0).sum()),'worse':int((x>0).sum()),'equal':int((x==0).sum())}

def main():
    tick=time.perf_counter();jobs=[];cases={};table=[];historical={}
    labels=[('pilot','dt_PV_layer19_whole_pilot_summary_20260909.json','codex_dt_PV_layer19_whole_pilot_20260909_v1')]
    labels += [(f's{i}',f'dt_PV_layer19_remaining_regression_summary_20260909_s{i}.json',f'codex_dt_PV_layer19_remaining_regression_20260909_s{i}_v1') for i in range(3)]
    protocols={};identity=('checkpoint','cache_paths','cache_hashes','native_model_sha256','installed_FA_interface_sha256',
        'checkpoint_config_tokenizer_sha256','expected_weight_stats','official_source_blob_sha1','span_source_sha256','finite_FA_library','finite_FA_library_sha256')
    for label,name,directory in labels:
        path=A/name;s=read(path);D=A/'snapshot/tmp'/directory;r=read(D/'results.json');p=read(D/'protocol.json')
        assert s['status']=='independent_PV19_whole_pilot_audit_passed'
        assert s['results_sha256']==sha(D/'results.json') and s['protocol_sha256']==sha(D/'protocol.json')
        assert s['vectors_sha256']==sha(D/'vectors.npz') and s['receipt_sha256']==sha(D/'terminal_receipt.json')
        assert r['protocol']==p and set(s['cases'])==set(p['quality_cases'])
        if protocols:
            for k in identity:assert p[k]==protocols['pilot'][k],k
        protocols[label]=p
        for key,c in s['cases'].items():
            assert key not in cases;cases[key]={'job':label,'summary':c,'input':r['cases'][key]['input'],'gold':r['cases'][key]['gold']}
        jobs.append({'label':label,'directory':str(D),'summary_path':str(path),'summary_sha256':sha(path),
            'results_sha256':s['results_sha256'],'protocol_sha256':s['protocol_sha256'],'vectors_sha256':s['vectors_sha256'],
            'wall_seconds':s['wall_seconds'],'counts':s['counts']})
    expected={f'{dataset}_{i}' for dataset in ('niah_mq_q2','morehopqa') for i in range(4)}
    assert set(cases)==expected
    for i in range(4):
        D=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{i}_v1'
        p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
        assert sha(D/'results.json')==receipt['files']['results.json']['sha256']
        assert sha(D/'vectors.npz')==r['vectors_sha256'] and r['protocol']==p
        assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
        assert r['FT_calls']==2 and r['scorer_returned']==168
        for k in identity:assert p[k]==protocols['pilot'][k],k
        z=np.load(D/'vectors.npz',allow_pickle=False)
        historical[str(i)]={'directory':str(D),'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),
            'vectors_sha256':sha(D/'vectors.npz'),'receipt_sha256':sha(D/'terminal_receipt.json'),'FT_commit':p['FT_commit']}
        for dataset in ('niah_mq_q2','morehopqa'):
            key=f'{dataset}_{i}';old=r['cases'][key];current=cases[key];s=current['summary']
            assert old['input']==current['input'] and old['gold']==current['gold']
            assert p['fixed_records'][key]['source_record_sha256']==protocols[current['job']]['fixed_records'][key]['source_record_sha256']
            ft={}
            for method in ('FT0','FT3'):
                curve=old['curves'][method]
                assert curve['status']=='complete' and curve['returned_forwards']==21
                nt=needle(z[key+'_'+method+'_evaluated'],old['input']['keep'],old['gold'],curve['needle'])
                ft[method]={'RISE':curve['return_metrics'][0],'MAS':curve['return_metrics'][1],'needle':nt,
                    'native_score_endpoints':[curve['scores'][j] for j in (0,20)],'historical_source_index':i}
            current['historical_FT']=ft
            current['current_control_minus_historical_FT_native_endpoints']={m:[s['methods']['control']['scores'][j]-ft[m]['native_score_endpoints'][pos] for pos,j in enumerate((0,20))] for m in ft}
    for dataset in ('niah_mq_q2','morehopqa'):
        for i in range(4):
            key=f'{dataset}_{i}';c=cases[key];s=c['summary']
            row={'case':key,'job':c['job'],'input_sha256':c['input']['input_sha256'],
                'methods':{m:{k:s['methods'][m][k] for k in ('RISE','MAS','needle')} for m in ('control','candidate')},
                'candidate_minus_control':s['candidate_minus_control'],'MAS_delta_ledger':s['candidate_minus_control_MAS_decomposition'],
                'same_control_error_summary':s['same_control_error_summary'],
                'fixed_control_interior_improved':len(s['interior_absolute_error_improved_steps']),
                'fixed_control_interior_worse':len(s['interior_absolute_error_worse_steps']),
                'historical_FT':c['historical_FT'],'current_control_minus_historical_FT_native_endpoints':c['current_control_minus_historical_FT_native_endpoints'],
                'same_case_native_root_drift':s['same_case_native_root_drift'],
                'same_case_root_seed_deltas':s['same_case_root_seed_deltas'],
                'candidate_minus_control_native_score_endpoints':s['candidate_minus_control_native_score_endpoints']}
            assert all(v['equal'] for v in row['same_case_native_root_drift'].values())
            assert all(v==0 for v in row['same_case_root_seed_deltas'].values())
            assert row['candidate_minus_control_native_score_endpoints']==[0,0]
            table.append(row)
    datasets={}
    for dataset in ('niah_mq_q2','morehopqa'):
        selected=[row for row in table if row['case'].startswith(dataset+'_')];assert len(selected)==4
        output={'case_count':4,'means':{},'candidate_vs_current_paired':{},'historical_FT_descriptive_comparisons':{}}
        for method in ('control','candidate','FT0','FT3'):
            values=[row['methods'][method] if method in ('control','candidate') else row['historical_FT'][method] for row in selected]
            output['means'][method]={k:float(np.mean([v[k] for v in values])) for k in ('RISE','MAS')}
            nts=[v['needle'] for v in values]
            if any(x is not None for x in nts):
                assert all(x is not None for x in nts);hits=sum(x['hits'] for x in nts);den=sum(x['eligible_gold_denominator'] for x in nts)
                output['means'][method]['needle']={'macro':float(np.mean([x['reported'] for x in nts])),
                    'total_hits':hits,'total_eligible_gold':den,'micro':hits/den}
        for metric in ('RISE','MAS'):
            delta=[row['candidate_minus_control'][metric] for row in selected]
            output['candidate_vs_current_paired'][metric]={'mean_delta':float(np.mean(delta)),**counts(delta)}
        output['candidate_vs_current_paired']['same_control_MAE_AUC']={'control_mean':float(np.mean([r['same_control_error_summary']['control']['MAE_AUC'] for r in selected])),
            'candidate_mean':float(np.mean([r['same_control_error_summary']['candidate']['MAE_AUC'] for r in selected])),
            **counts([r['same_control_error_summary']['candidate']['MAE_AUC']-r['same_control_error_summary']['control']['MAE_AUC'] for r in selected])}
        if dataset=='niah_mq_q2':output['candidate_vs_current_paired']['needle']=counts([r['candidate_minus_control']['needle'] for r in selected],lower=False)
        output['MAS_mean_delta_ledger']={k:float(np.mean([r['MAS_delta_ledger'][k] for r in selected])) for k in selected[0]['MAS_delta_ledger']}
        close(sum(output['MAS_mean_delta_ledger'].values()),output['candidate_vs_current_paired']['MAS']['mean_delta'])
        for method in ('FT0','FT3'):
            output['historical_FT_descriptive_comparisons'][method]={metric:{'candidate_mean_minus_historical_mean':output['means']['candidate'][metric]-output['means'][method][metric],
                **counts([r['methods']['candidate'][metric]-r['historical_FT'][method][metric] for r in selected])} for metric in ('RISE','MAS')}
        datasets[dataset]=output
    total_counts={k:sum(j['counts'][k] for j in jobs) for k in jobs[0]['counts']}
    assert total_counts['DT']==16 and total_counts['finite_FLA']==400 and total_counts['finite_FA']==total_counts['public_FA_LSE']==128
    assert total_counts['original_score_forwards']==336
    out={'status':'fixed_eight_used_case_PV19_independent_aggregation_complete','analyzer_sha256':sha(__file__),
        'jobs':jobs,'historical_FT_source_jobs':historical,'case_table':table,'datasets':datasets,
        'overall_case_outcomes':{metric:counts([r['candidate_minus_control'][metric] for r in table]) for metric in ('MAS','RISE')},
        'fixed_control_interiors':{'improved':sum(r['fixed_control_interior_improved'] for r in table),'worse':sum(r['fixed_control_interior_worse'] for r in table),'total':8*19},
        'total_counts':total_counts,'total_wall_seconds':sum(j['wall_seconds'] for j in jobs),
        'scope':'All eight fixed previously used author cases are included; no selected-case omission, new benchmark, FT call, generation, or GPU operation by this aggregate. NI and MH metric means remain separate. Same-control-mask magnitude gains do not establish arbitrary-coalition accuracy or unchanged ranking.',
        'historical_FT_scope':'FT0/FT3 are unchanged original e81 Both outputs from the original eight-case runs, same input/target/gold and native source/weight identity, but a different process. Full clean/EOS score deltas are listed per case. Endpoint equality alone cannot prove all intermediate historical scorer outputs identical. FT explains its original sink representation; DT targets full fixed-answer logprob. No historical score is substituted into a current DT curve.',
        'cost_scope':'Four fresh jobs each have one model load and one eager NI0 initialization. These16 attribute calls retain first-shape compilation/order effects; same-case paired peaks are equal per independent audit, but no warmed speed claim. Separate subsequently run cost work is not included.',
        'decision_scope':'MAS improves6/8 and worsens2/8; RISE improves2/8 and worsens6/8. NI needle loses2 hits overall; current factory is not automatically promoted. Remaining negative-density alignment error and ranking tradeoffs prevent a universal-improvement or completed-repair claim. Goal remains active.',
        'audit_seconds':time.perf_counter()-tick}
    target=A/'dt_PV_layer19_eight_case_summary_20260909.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'sha256':sha(target),'outcomes':out['overall_case_outcomes'],'datasets':datasets,'seconds':out['total_wall_seconds']}))

if __name__=='__main__':main()
