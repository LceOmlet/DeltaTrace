"""Verify every row/vector and apply the predeclared short-B1 acceptance gate."""
import argparse,csv,hashlib,io,json
from pathlib import Path
import numpy as np


def summarize(raw,protocol_path,run_name):
    sha=lambda b:hashlib.sha256(b).hexdigest()
    protocol=json.loads(protocol_path.read_bytes());queue=json.loads((raw/run_name/'queue.json').read_bytes())
    assert queue['status']=='complete'
    expected=[]
    for ri,lengths in enumerate(protocol['length_orders'],1):
        for index,n in enumerate(lengths):
            order=protocol['methods'] if (index+ri)%2 else list(reversed(protocol['methods']))
            expected.extend(f'r{ri}_{m}_{n}' for m in order)
    assert len(expected)==24
    assert [j['name'] for j in queue['jobs']]==expected
    cells=[];reports={};inputs={};all_rows=[];verification=[]
    for job in queue['jobs']:
        folder=raw/run_name/job['name'];data=json.loads((folder/'results.json').read_bytes())
        assert data['status']=='complete' and data['method']==job['method'] and data['family']=='qwen3'
        assert data['driver_sha256']==protocol['driver_sha256'] and data['protocol_sha256']==sha(protocol_path.read_bytes())
        assert len(data['cases'])==1 and len(data['rows'])==5 and all(r['status']=='ok' for r in data['rows'])
        reports[job['name']]={'model_load_seconds':data['model_load_seconds'],'initialization':data['initialization'],
            'raw_sha256':sha((folder/'results.json').read_bytes())}
        is_dt=job['method'].startswith('deltatrace_')
        with np.load(folder/'vectors.npz') as vectors:
            for case in data['cases']:
                n=case['input_length'];assert n==job['input_length'];ids=np.asarray(case['input_ids'],dtype=np.int64)
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
    methods=protocol['methods'];dt=methods[0];rng=np.random.default_rng(protocol['acceptance_frozen_before_measurement']['bootstrap']['seed'])
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
    import math
    costs=[];processes=[];rows=[]
    for job in json.loads((raw/run_name/'queue.json').read_bytes())['jobs']:
        data=json.loads((raw/run_name/job['name']/'results.json').read_bytes());group=data['rows'];n=job['input_length']
        setup=sum(i['seconds'] for i in data['initialization']);warm=[r for r in group if r['phase']=='warm'];steady=[r for r in group if r['phase']=='measured'];first=warm[0]
        init_memory=data['pre_call_initialization_memory'];load_memory=data['model_load_memory']
        processes.append({'job':job['name'],'round':job['round'],'method':job['method'],'target_input_tokens':n,
            'model_load_seconds':data['model_load_seconds'],'controller_setup_seconds':setup,
            'all_runner_constructions_seconds':sum(r['runner_init_seconds'] for r in group),
            'model_load_memory':load_memory,'pre_call_initialization_memory':init_memory,
            'close':data.get('close'),'process_elapsed_seconds':job['ended']-job['started']})
        costs.append({'job':job['name'],'round':job['round'],'method':job['method'],'target_input_tokens':n,'actual_total_tokens':first['actual_total_tokens'],
            'model_load_seconds':data['model_load_seconds'],'controller_setup_seconds':setup,
            'first_geometry_seconds':first['time_sec'],'second_warm_seconds':warm[1]['time_sec'],
            'first_geometry_peak_allocated_gb':first['peak_allocated_gb'],'first_geometry_peak_reserved_gb':first['peak_mem_reserved_gb'],
            'cold_warm_peak_allocated_gb':max(r['peak_allocated_gb'] for r in group),
            'cold_warm_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in group),
            'all_cost_peak_allocated_gb':max(init_memory['peak_allocated_gb'],load_memory['peak_allocated_gb'],*(r['peak_allocated_gb'] for r in group)),
            'all_cost_peak_reserved_gb':max(init_memory['peak_reserved_gb'],load_memory['peak_reserved_gb'],*(r['peak_mem_reserved_gb'] for r in group)),
            'steady_seconds':sum(r['time_sec'] for r in steady)/3,
            'steady_peak_allocated_gb':max(r['peak_allocated_gb'] for r in steady),'steady_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in steady),
            'resident_before_allocated_gb':max(r['resident_before']['allocated_bytes']/1e9 for r in steady),
            'resident_before_reserved_gb':max(r['resident_before']['reserved_bytes']/1e9 for r in steady),
            'resident_after_allocated_gb':max(r['resident_after']['allocated_bytes']/1e9 for r in steady),
            'resident_after_reserved_gb':max(r['resident_after']['reserved_bytes']/1e9 for r in steady),
            'host_current_RSS_max_gb':max(r['process_memory']['rss_current_bytes']/1e9 for r in group),
            'host_process_lifetime_RSS_highwater_gb':max(load_memory['host_process_highwater_bytes']/1e9,*(r['process_memory']['rss_process_highwater_bytes']/1e9 for r in group)),
            'cold_runtime_cost':first['runtime_cost'],'steady_runtime_costs':[r['runtime_cost'] for r in steady]})
        rows.extend(dict(r,job=job['name'],round=job['round']) for r in group)
    memory=[];amortization=[]
    for n in [128,256,512,1024]:
        select=lambda method:[r for r in costs if r['method']==method and r['target_input_tokens']==n]
        dt=select('deltatrace_streamed');assert len(dt)==2
        for method in ['ifr_multi_hop_both','ifr_multi_hop']:
            ft=select(method);assert len(ft)==2
            row={'target_input_tokens':n,'FT_method':method,'scope':'Both independently initialized geometry processes; model load/setup plus every cold/warm/measured call. Zero exclusions.'}
            for key in ['allocated','reserved']:
                field='all_cost_peak_'+key+'_gb';a=max(c[field] for c in dt);b=max(c[field] for c in ft)
                row.update({f'DT_peak_{key}_gb':a,f'FT_peak_{key}_gb':b,f'DT_{key}_below_FT':a<b,f'{key}_reduction_percent':100*(1-a/b)})
                row[key+'_round_ratios']=[next(c[field] for c in dt if c['round']==ri)/next(c[field] for c in ft if c['round']==ri) for ri in [1,2]]
                row['steady_'+key+'_below_FT']=max(c['steady_peak_'+key+'_gb'] for c in dt)<max(c['steady_peak_'+key+'_gb'] for c in ft)
            row['acceptance_passed']=row['DT_allocated_below_FT'] and row['DT_reserved_below_FT'];memory.append(row)
            for ri in [1,2]:
                d=next(c for c in dt if c['round']==ri);f=next(c for c in ft if c['round']==ri)
                saving=f['steady_seconds']-d['steady_seconds'];extra=d['first_geometry_seconds']-f['first_geometry_seconds']
                total_extra=extra+d['model_load_seconds']+d['controller_setup_seconds']-f['model_load_seconds']-f['controller_setup_seconds']
                count=lambda e:max(1,math.floor(e/saving)+2) if saving>0 else None
                amortization.append({'target_input_tokens':n,'round':ri,'FT_method':method,'FT_warm_minus_DT_warm_seconds':saving,
                    'DT_first_minus_FT_first_seconds':extra,'estimated_crossover_calls_without_load_and_setup':count(extra),
                    'estimated_crossover_calls_with_load_and_setup':count(total_extra)})
    assert len(rows)==120 and len(costs)==len(processes)==24
    summary.update(cost_cells=costs,process_costs=processes,memory_comparisons=memory,cold_amortization=amortization,
        memory_acceptance_passed=all(r['acceptance_passed'] for r in memory),
        acceptance_scope='Four fixed-output32 short-input bins only: complete warm API time with two independent rounds and95% bootstrap, plus independent fixed-geometry capacity including model load, setup and all cold/warm peaks. Longer-rollout capacity and cold shape transitions have separate evidence; no universal memory claim.',
        previous_v1_transition_gate='The original descending mixed-geometry v1 gate failed and remains unchanged in memory_confirmation_summary.json and its raw archive. This independent-geometry protocol was frozen before v2 measurement and retained unchanged for v3.',
        cost_accounting={'unit':'Decimal GB. Allocator counters in the MetaX CUDA-compatible runtime; not whole-device mx-smi readings.',
            'wall':'Original synchronized author timer including complete original model root, input copies, finite propagation/native recomputation/SAC, state checks, original predicates and full CPU return/destruction. Cold also includes2 complete model+finite warmups,1 capture and1 graph replay.',
            'graph_memory':'Warm allocated excludes graph-pool temporaries; both physical reserved and cold allocated are required and reported. Initializer peaks are separately recorded and included in capacity acceptance.',
            'host_memory':'Current RSS sampled after calls; process highwater is lifetime, not per-call. Model-load host highwater is separately recorded. No host-memory superiority claim.',
            'amortization':'T(K)=load+setup+first+(K-1)*observed_mean_warm. Crossover counts are estimates assuming unchanged future cost, not measured guarantees.',
            'host_sections':'Host wall sections can overlap queued GPU work and are already included in complete wall time; do not add them again or call them GPU kernel durations.',
            'API_calls':{'timed':120,'warm_rows':48,'measured_rows':72,'separate_output_audits':24,'separate_original_retained_references':8,'total_complete_calls':152},
            'new_generation_calls':0,'new_metric_calls':0},
        all_current_inputs_and_full_vectors_verified=True,
        baseline_return_limitation='Untouched FT Both may return nonfinite matrix elements, counted per cell. Timing follows its original complete API; no sanitization, row exclusion or FT numerical-validity claim.')
    summary['acceptance_passed']=summary['speed_acceptance_passed'] and summary['memory_acceptance_passed']
    target=raw/run_name/'verified_all_rows.csv'
    flat=[{k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--run',default='memory_v3_confirmation');p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    summary,csv_text=summarize(a.raw,a.protocol,a.run)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',newline='\n')
    a.output_prefix.with_suffix('.csv').write_text(csv_text,newline='\n')
    print(json.dumps({'acceptance_passed':summary['acceptance_passed'],'verified_cells':len(summary['verified_cells']),'rows':summary['all_timed_rows_count'],'comparisons':summary['comparisons'],'memory_comparisons':summary['memory_comparisons']},indent=2))
