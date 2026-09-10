"""Verify every row/vector and apply the predeclared short-B1 acceptance gate."""
import argparse,csv,hashlib,io,json
from pathlib import Path
import numpy as np


def summarize(raw,protocol_path,run_name):
    sha=lambda b:hashlib.sha256(b).hexdigest()
    protocol=json.loads(protocol_path.read_bytes());queue=json.loads((raw/run_name/'queue.json').read_bytes())
    assert queue['status']=='complete'
    expected=[f'round{r+1}_{m}' for r,methods in enumerate(protocol['method_rounds']) for m in methods]
    assert [j['name'] for j in queue['jobs']]==expected
    cells=[];reports={};inputs={};all_rows=[];verification=[]
    for job in queue['jobs']:
        folder=raw/run_name/job['name'];data=json.loads((folder/'results.json').read_bytes())
        assert data['status']=='complete' and data['method']==job['method'] and data['family']=='qwen3'
        assert data['driver_sha256']==protocol['driver_sha256'] and data['protocol_sha256']==sha(protocol_path.read_bytes())
        assert len(data['rows'])==20 and all(r['status']=='ok' for r in data['rows'])
        reports[job['name']]={'model_load_seconds':data['model_load_seconds'],'initialization':data['initialization'],
            'raw_sha256':sha((folder/'results.json').read_bytes())}
        is_dt=job['method'].startswith('deltatrace_')
        with np.load(folder/'vectors.npz') as vectors:
            for case in data['cases']:
                n=case['input_length'];ids=np.asarray(case['input_ids'],dtype=np.int64)
                assert len(ids)==case['lengths']['total_tokens']<=1024
                assert sha(ids.tobytes())==case['input_sha256']
                if n in inputs:assert np.array_equal(inputs[n],ids)
                else:inputs[n]=ids
                vec=vectors[str(n)];audit=case['audit']
                assert sha(vec.tobytes())==audit['vector_sha256']
                finite=bool(np.isfinite(vec).all());assert finite==audit['finite']
                if is_dt:assert finite
                assert list(vec.shape)==audit['vector_shape']
                if is_dt:
                    assert np.array_equal(vec,vectors[str(n)+'_root']) and np.array_equal(vec,vectors[str(n)+'_retained'])
                    assert case['comparisons']['retained_same_process']['exact'] and audit['full_vector_exact_to_retained_same_process']
                    assert case['math_diagnostics_exact']
                    detail=case['details'];reference=case['retained_details']
                    for key in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']:assert detail[key]==reference[key]
                    assert detail['native_layer_replay_calls']==0 and detail['extra_native_fa_attention_calls']==36
                    assert len(detail['layer_checks'])==36 and len(detail['public_FA_capture_checks'])==36
                    assert all(c['public_output_exact_to_actual_model_FA'] for c in detail['public_FA_capture_checks'])
                    assert all(c['native_input_exact'] and c['native_output_exact'] for c in detail['native_layer_boundary_checks']['paired_root'])
                    assert detail['root_retention_mutation_audit']=={'enabled':True,'predicates':720}
                    if 'native_graph_execution' in detail:
                        execution=detail['native_graph_execution'];build=case['graph_build_info']
                        assert detail['deferred_validation']['predicates']==829
                        assert execution['graph_replays']==1 and execution['geometry_cache_entries']==1
                        assert execution['static_input_tensors']==883 and execution['static_input_storage_copies']==658
                        assert execution['every_input_tensor_refreshed'] and not execution['attribution_results_reused']
                        assert execution['native_model_root_calls']==1 and execution['native_model_root_endpoint_batch']==2
                        assert execution['build_this_call'] is None and execution['finite_program_warmups_this_call']==0
                        assert detail['graph_input_copy_audit']=={'enabled':True,'predicates':883,'all_passed':True}
                        assert build['native_graph_program_recordings']==1 and len(build['warm_finite_programs'])==2
                        assert all(w['strict_predicates']==829 and w['all_passed'] for w in build['warm_finite_programs'])
                    else:
                        assert detail['deferred_validation']['predicates']==1548
                        assert all(a['compiled_input_buffers_exact']==[True]*10 for a in detail['finite_attention_activity'])
                    assert detail['deferred_validation']['all_passed']
                    assert detail['deferred_validation']['statistics']==36
                    assert len(audit['actual_calls'])==len(case['retained_actual_calls'])==1
                    assert audit['actual_calls']==case['retained_actual_calls']
                rows=[r for r in data['rows'] if r['target_input_tokens']==n]
                warm=[r for r in rows if r['phase']=='warm'];measured=[r for r in rows if r['phase']=='measured']
                assert len(warm)==2 and len(measured)==3 and len(rows)==5
                assert all(r['input_sha256']==case['input_sha256'] for r in rows)
                times=[r['time_sec'] for r in measured]
                cells.append({'round':job['round'],'method':job['method'],'target_input_tokens':n,
                    'actual_total_tokens':len(ids),'input_sha256':case['input_sha256'],'times_seconds':times,
                    'full_return_vector_finite':finite,'nonfinite_return_elements':int(np.sum(~np.isfinite(vec))),
                    'mean_seconds':float(np.mean(times)),'median_seconds':float(np.median(times)),
                    'min_seconds':min(times),'max_seconds':max(times),'warm_seconds':[r['time_sec'] for r in warm],
                    'warm_peak_allocated_gb':[r['peak_allocated_gb'] for r in warm],
                    'graph_build_info':case.get('graph_build_info'),
                    'peak_allocated_gb':max(r['peak_allocated_gb'] for r in measured),
                    'peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in measured),
                    'output_audit_seconds':audit['separately_charged_seconds'],
                    'retained_reference_seconds':case.get('retained_comparison_seconds',0),
                    'historical_comparison':case.get('comparisons',{}).get('historical_other_cache')})
                all_rows.extend(rows);verification.append({'job':job['name'],'input_length':n,'verified':True})
    methods=protocol['method_rounds'][0];dt=methods[0];rng=np.random.default_rng(protocol['acceptance_frozen_before_measurement']['bootstrap']['seed'])
    resamples=protocol['acceptance_frozen_before_measurement']['bootstrap']['resamples'];comparisons=[];pooled=[]
    for n in protocol['input_lengths']:
        arrays={}
        for method in methods:
            selected=[c for c in cells if c['method']==method and c['target_input_tokens']==n]
            assert len(selected)==2
            values=np.asarray([t for c in selected for t in c['times_seconds']]);assert len(values)==6
            arrays[method]=values
            pooled.append({'method':method,'target_input_tokens':n,'actual_total_tokens':selected[0]['actual_total_tokens'],
                'times_seconds':values.tolist(),'mean_seconds':float(values.mean()),'median_seconds':float(np.median(values)),
                'min_seconds':float(values.min()),'max_seconds':float(values.max()),
                'peak_allocated_gb':max(c['peak_allocated_gb'] for c in selected),'peak_reserved_gb':max(c['peak_reserved_gb'] for c in selected)})
        for method in methods[1:]:
            a=arrays[dt];b=arrays[method];ratio=float(a.mean()/b.mean())
            sampled_a=a[rng.integers(0,6,(resamples,6))].mean(1);sampled_b=b[rng.integers(0,6,(resamples,6))].mean(1)
            ci=np.percentile(sampled_a/sampled_b,[2.5,97.5]).tolist()
            round_ratios=[]
            for r in [1,2]:
                first=next(c for c in cells if c['round']==r and c['method']==dt and c['target_input_tokens']==n)
                second=next(c for c in cells if c['round']==r and c['method']==method and c['target_input_tokens']==n)
                round_ratios.append(first['mean_seconds']/second['mean_seconds'])
            passed=ratio<1 and max(round_ratios)<1 and ci[1]<1
            comparisons.append({'target_input_tokens':n,'FT_method':method,'DT_to_FT_mean_ratio':ratio,
                'latency_reduction_percent':100*(1-ratio),'round_mean_ratios':round_ratios,
                'bootstrap_ratio_95_percentile_interval':ci,'acceptance_passed':passed})
    summary={'version':protocol['version'],'protocol_sha256':sha(protocol_path.read_bytes()),'reports':reports,
        'verified_cells':verification,'cells':cells,'pooled_cells':pooled,'comparisons':comparisons,
        'acceptance_passed':all(c['acceptance_passed'] for c in comparisons),
        'all_timed_rows_count':len(all_rows),'warm_rows_count':sum(r['phase']=='warm' for r in all_rows),
        'measured_rows_count':sum(r['phase']=='measured' for r in all_rows),'excluded_timing_rows':0,
        'timed_seconds':sum(r['time_sec'] for r in all_rows),'audit_seconds':sum(c['output_audit_seconds']+c['retained_reference_seconds'] for c in cells),
        'scope':'Two prespecified serial rounds on this environment and fixed author exp1 fallback inputs, not a guarantee for every workload or device.'}
    csv_file=io.StringIO(newline='');fields=['method','target_input_tokens','actual_total_tokens','mean_seconds','median_seconds','min_seconds','max_seconds','peak_allocated_gb','peak_reserved_gb']
    writer=csv.DictWriter(csv_file,fieldnames=fields,lineterminator='\n');writer.writeheader()
    writer.writerows({k:c[k] for k in fields} for c in pooled)
    return summary,csv_file.getvalue()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    summary,csv_text=summarize(a.raw,a.protocol,a.run)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',newline='\n')
    a.output_prefix.with_suffix('.csv').write_text(csv_text,newline='\n')
    print(json.dumps({'acceptance_passed':summary['acceptance_passed'],'verified_cells':len(summary['verified_cells']),'rows':summary['all_timed_rows_count'],'comparisons':summary['comparisons']},indent=2))
