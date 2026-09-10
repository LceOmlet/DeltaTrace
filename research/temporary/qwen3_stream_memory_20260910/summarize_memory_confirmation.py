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
                    assert len(detail['layer_checks'])==36 and len(detail['public_FA_capture_checks'])==36
                    assert all(c['public_output_exact_to_actual_model_FA'] for c in detail['public_FA_capture_checks'])
                    execution=detail['native_graph_execution'];build=case['graph_build_info']
                    assert detail['deferred_validation']['all_passed'] and detail['deferred_validation']['statistics']==36
                    assert execution['graph_replays']==1 and execution['geometry_cache_entries']==1
                    assert execution['every_input_tensor_refreshed'] and not execution['attribution_results_reused']
                    assert execution['build_this_call'] is None and execution['finite_program_warmups_this_call']==0
                    assert build['native_graph_program_recordings']==1 and len(build['warm_finite_programs'])==2
                    assert len(case['retained_actual_calls'])==1
                    if job['method']=='deltatrace_streamed':
                        predicates=829+(36 if detail['native_projection_SAC']['selected_projections'] else 0)
                        assert detail['native_layer_replay_calls']==detail['extra_native_fa_attention_calls']==36
                        assert detail['deferred_validation']['predicates']==predicates
                        assert all(c['native_input_exact'] and c['native_output_exact'] for c in detail['native_layer_boundary_checks']['paired_batch'])
                        assert execution['static_input_tensors']==execution['static_input_storage_copies']==1
                        assert execution['native_model_Python_root_calls']==0 and execution['native_model_GPU_root_executions']==1
                        assert detail['graph_input_copy_audit']=={'enabled':True,'predicates':1,'all_passed':True}
                        assert all(w['strict_predicates']==predicates and w['all_passed'] for w in build['warm_finite_programs'])
                        assert not audit['actual_calls'] and case['graph_input_matches_original_root']
                        assert detail['whole_root_graph']['fresh_graph_input_sha256']==case['retained_actual_calls'][0]['input_sha256']
                        assert build['original_model_Python_calls_during_build']==3
                        assert build['root_and_finite_captured_together']
                    else:
                        assert detail['native_layer_replay_calls']==0 and detail['extra_native_fa_attention_calls']==36
                        assert detail['deferred_validation']['predicates']==829
                        assert all(c['native_input_exact'] and c['native_output_exact'] for c in detail['native_layer_boundary_checks']['paired_root'])
                        assert detail['root_retention_mutation_audit']=={'enabled':True,'predicates':720}
                        assert execution['static_input_tensors']==883 and execution['static_input_storage_copies']==658
                        assert execution['native_model_root_calls']==1 and execution['native_model_root_endpoint_batch']==2
                        assert detail['graph_input_copy_audit']=={'enabled':True,'predicates':883,'all_passed':True}
                        assert all(w['strict_predicates']==829 and w['all_passed'] for w in build['warm_finite_programs'])
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
        'speed_acceptance_passed':all(c['acceptance_passed'] for c in comparisons if c['FT_method'].startswith('ifr_')),
        'old_DT_comparison_required_for_speed_gate':False,
        'all_timed_rows_count':len(all_rows),'warm_rows_count':sum(r['phase']=='warm' for r in all_rows),
        'measured_rows_count':sum(r['phase']=='measured' for r in all_rows),'excluded_timing_rows':0,
        'timed_seconds':sum(r['time_sec'] for r in all_rows),'audit_seconds':sum(c['output_audit_seconds']+c['retained_reference_seconds'] for c in cells),
        'scope':'Two prespecified serial rounds on this environment and fixed author exp1 fallback inputs, not a guarantee for every workload or device.'}
    enrich_costs(summary,raw,run_name)
    csv_file=io.StringIO(newline='');fields=['method','target_input_tokens','actual_total_tokens','mean_seconds','median_seconds','min_seconds','max_seconds','peak_allocated_gb','peak_reserved_gb']
    writer=csv.DictWriter(csv_file,fieldnames=fields,lineterminator='\n');writer.writeheader()
    writer.writerows({k:c[k] for k in fields} for c in pooled)
    return summary,csv_file.getvalue()



def enrich_costs(summary,raw,run_name):
    costs=[];rows=[];processes=[]
    for job in json.loads((raw/run_name/'queue.json').read_bytes())['jobs']:
        data=json.loads((raw/run_name/job['name']/'results.json').read_bytes())
        setup=sum(i['seconds'] for i in data['initialization'])
        processes.append({'job':job['name'],'round':job['round'],'method':job['method'],
            'model_load_seconds':data['model_load_seconds'],'controller_setup_seconds':setup,
            'all_runner_constructions_seconds':sum(r['runner_init_seconds'] for r in data['rows']),
            'close':data.get('close'),'process_elapsed_seconds':job['ended']-job['started']})
        for n in [128,256,512,1024]:
            group=[r for r in data['rows'] if r['target_input_tokens']==n]
            warm=[r for r in group if r['phase']=='warm'];steady=[r for r in group if r['phase']=='measured']
            first=warm[0]
            costs.append({'job':job['name'],'round':job['round'],'method':job['method'],'target_input_tokens':n,
                'actual_total_tokens':first['actual_total_tokens'],'first_geometry_seconds':first['time_sec'],
                'first_geometry_peak_allocated_gb':first['peak_allocated_gb'],'first_geometry_peak_reserved_gb':first['peak_mem_reserved_gb'],
                'second_warm_seconds':warm[1]['time_sec'],
                'steady_seconds':sum(r['time_sec'] for r in steady)/3,
                'steady_peak_allocated_gb':max(r['peak_allocated_gb'] for r in steady),
                'steady_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in steady),
                'resident_before_allocated_gb':max(r['resident_before']['allocated_bytes']/1e9 for r in steady),
                'resident_after_allocated_gb':max(r['resident_after']['allocated_bytes']/1e9 for r in steady),
                'resident_after_reserved_gb':max(r['resident_after']['reserved_bytes']/1e9 for r in steady),
                'host_current_RSS_max_gb':max(r['process_memory']['rss_current_bytes']/1e9 for r in steady),
                'host_process_lifetime_RSS_highwater_gb':max(r['process_memory']['rss_process_highwater_bytes']/1e9 for r in group),
                'cold_runtime_cost':first['runtime_cost'],
                'steady_runtime_costs':[r['runtime_cost'] for r in steady]})
            for r in group:rows.append(dict(r,job=job['name'],round=job['round']))
    memory=[];amortization=[]
    for n in [128,256,512,1024]:
        select=lambda m:[c for c in costs if c['method']==m and c['target_input_tokens']==n]
        dt=select('deltatrace_streamed');old=select('deltatrace_graphed')
        new_peak=max(c['steady_peak_allocated_gb'] for c in dt);old_peak=max(c['steady_peak_allocated_gb'] for c in old)
        reduction=1-new_peak/old_peak
        memory.append({'target_input_tokens':n,'old_DT_steady_peak_allocated_gb':old_peak,
            'new_DT_steady_peak_allocated_gb':new_peak,'absolute_reduction_gb':old_peak-new_peak,
            'reduction_percent':100*reduction,'acceptance_passed':reduction>=(.2 if n==1024 else .05)})
        for method in ['ifr_multi_hop_both','ifr_multi_hop']:
            ft=select(method)
            for d,b in zip(dt,ft):
                assert d['round']==b['round']
                saving=b['steady_seconds']-d['steady_seconds'];extra=d['first_geometry_seconds']-b['first_geometry_seconds']
                import math
                count=max(1,math.floor(extra/saving)+2) if saving>0 else None
                amortization.append({'target_input_tokens':n,'round':d['round'],'FT_method':method,
                    'DT_first_minus_FT_first_seconds':extra,'FT_warm_minus_DT_warm_seconds':saving,
                    'estimated_first_strictly_faster_call_count_excluding_model_and_setup':count})
    summary.update(cost_cells=costs,process_costs=processes,memory_comparisons=memory,
        memory_acceptance_passed=all(c['acceptance_passed'] for c in memory),cold_amortization=amortization,
        cost_accounting={'unit':'GB = 10^9 bytes, ms = 1000 seconds. CUDA allocator numbers on MetaX runtime; not whole-device mx-smi memory.',
            'complete_wall':'Original synchronized author timer, includes fresh model root, all copies, graph replay, all original predicates, CPU full-vector/scalar conversion and output destruction. Cold also includes 2 finite warmups and graph capture.',
            'host_sections':'CPU wall sections overlap queued GPU work. Layer-copy host time is already included in root time; do not add twice or interpret as GPU kernel duration.',
            'cold_scope':'First call at each geometry. Round1 starts empty method-specific compiler caches. Round2 reuses only same-method disk caches but has a fresh process and graph.',
            'resident':'Allocated/reserved before and after complete calls; process host RSS current sampled after call. ru_maxrss is lifetime highwater, NOT per-call peak.',
            'amortization_formula':'Stable geometry: T(K)=T_model_load+T_controller_setup+T_first+(K-1)*T_warm. Tabulated crossover compares T_first and T_warm only, assumes future warm costs equal observed means; not a measured guarantee.',
            'API_calls':{'timed':160,'separate_output_audits':32,'separate_original_retained_references':16,'total_complete_calls':208},
            'new_generation_calls':0,'new_metric_calls':0})
    summary['acceptance_passed']=summary['speed_acceptance_passed'] and summary['memory_acceptance_passed']
    goal_memory=[]
    for n in [128,256,512,1024]:
        select=lambda method:[r for r in rows if r['attr_func']==method and r['target_input_tokens']==n]
        dt=select('deltatrace_streamed')
        for method in ['ifr_multi_hop_both','ifr_multi_hop']:
            ft=select(method)
            comparison={'target_input_tokens':n,'FT_method':method,'scope':'All five cold/warm/measured calls and both process rounds; no exclusions.'}
            for key,field in [('allocated','peak_allocated_gb'),('reserved','peak_mem_reserved_gb')]:
                d=max(r[field] for r in dt);b=max(r[field] for r in ft)
                comparison.update({f'DT_peak_{key}_gb':d,f'FT_peak_{key}_gb':b,f'DT_{key}_below_FT':d<b,f'{key}_reduction_percent':100*(1-d/b)})
                comparison[f'{key}_round_ratios']=[max(r[field] for r in dt if r['round']==ri)/max(r[field] for r in ft if r['round']==ri) for ri in [1,2]]
            comparison['acceptance_passed']=comparison['DT_allocated_below_FT'] and comparison['DT_reserved_below_FT']
            goal_memory.append(comparison)
    summary['memory_comparisons_against_previous_DT']=summary.pop('memory_comparisons')
    summary['memory_comparisons']=goal_memory
    summary['memory_acceptance_passed']=all(c['acceptance_passed'] for c in goal_memory)
    summary['acceptance_passed']=summary['speed_acceptance_passed'] and summary['memory_acceptance_passed']
    summary['new_user_goal_memory_against_FT']=goal_memory
    summary['new_user_goal_short_input_gate_passed']=summary['acceptance_passed']
    summary['acceptance_scope']='Frozen short-input comparison only: warm complete-call speed against both original FT variants; cold plus warm/timed maximum allocated and reserved memory against both. Author compatibility, rollout study and delivery are separate requirements.'
    summary['cost_accounting']['complete_wall']='Original synchronized author timer, includes fresh original native model GPU execution, all input copies and model-state checks, finite rules and native recomputation/SAC, all original predicates, CPU full-vector/scalar return and destruction. Cold additionally includes two complete model+finite warmups and one capture plus replay. New hot graph executes original model GPU kernels without invoking Python model.forward.'
    summary['cost_accounting']['graph_memory']='Warm graph memory_allocated may omit temporary buffers held by its private pool. Acceptance uses maximum allocated AND reserved over all cold/warm/measured calls, including build warmup and graph recording; host process RSS highwater is lifetime, not per-call.'
    target=raw/run_name/'verified_all_rows.csv'
    flat=[]
    for r in rows:flat.append({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    summary,csv_text=summarize(a.raw,a.protocol,a.run)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',newline='\n')
    a.output_prefix.with_suffix('.csv').write_text(csv_text,newline='\n')
    print(json.dumps({'acceptance_passed':summary['acceptance_passed'],'verified_cells':len(summary['verified_cells']),'rows':summary['all_timed_rows_count'],'comparisons':summary['comparisons']},indent=2))
