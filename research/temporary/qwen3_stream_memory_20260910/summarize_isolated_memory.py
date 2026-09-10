"""Keep the failed transition gate and verify independent capacity explicitly."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics
import numpy as np
HERE=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def summarize(raw):
    protocol_path=HERE/'isolated_memory_protocol.json';protocol=json.loads(protocol_path.read_bytes())
    queue=json.loads((raw/'isolated_memory/queue.json').read_bytes());assert queue['status']=='complete' and len(queue['jobs'])==12
    speed=json.loads((HERE/'memory_confirmation_summary.json').read_bytes())
    assert speed['speed_acceptance_passed']
    assert not speed['memory_acceptance_passed'], 'The actual mixed-geometry transition failure must remain visible.'
    cells=[];inputs={};all_rows=[]
    for job in queue['jobs']:
        folder=raw/'isolated_memory'/job['name'];data=json.loads((folder/'results.json').read_bytes())
        assert data['status']=='complete' and data['driver_sha256']==protocol['driver_sha256'] and data['protocol_sha256']==sha(protocol_path.read_bytes())
        rows=data['rows'];assert len(rows)==5 and all(r['status']=='ok' for r in rows)
        case=data['cases'][0];n=job['input_length'];ids=np.asarray(case['input_ids'],dtype=np.int64)
        assert sha(ids.tobytes())==case['input_sha256'] and len(ids)==case['lengths']['total_tokens']
        if n in inputs:assert inputs[n]==case['input_sha256']
        else:inputs[n]=case['input_sha256']
        with np.load(folder/'vectors.npz') as z:
            vec=z[str(n)];assert sha(vec.tobytes())==case['audit']['vector_sha256']
            if job['method']=='deltatrace_streamed':
                assert np.array_equal(vec,z[str(n)+'_retained']) and np.isfinite(vec).all()
                assert case['math_diagnostics_exact'] and case['graph_input_matches_original_root']
                detail=case['details'];assert detail['deferred_validation']['all_passed']
                for key in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']:assert detail[key]==case['retained_details'][key]
        measured=[r for r in rows if r['phase']=='measured'];assert len(measured)==3
        cells.append({'method':job['method'],'input_length':n,'actual_total_tokens':len(ids),'input_sha256':case['input_sha256'],
            'first_complete_call_seconds':rows[0]['time_sec'],'second_warm_seconds':rows[1]['time_sec'],
            'descriptive_measured_mean_seconds':statistics.mean(r['time_sec'] for r in measured),
            'peak_allocated_gb':max(r['peak_allocated_gb'] for r in rows),'peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in rows),
            'steady_peak_allocated_gb':max(r['peak_allocated_gb'] for r in measured),'steady_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in measured),
            'model_load_seconds':data['model_load_seconds'],'setup_seconds':sum(x['seconds'] for x in data['initialization']),
            'runner_construction_seconds':[r['runner_init_seconds'] for r in rows],
            'resident_after_allocated_gb':max(r['resident_after']['allocated_bytes']/1e9 for r in measured),
            'resident_after_reserved_gb':max(r['resident_after']['reserved_bytes']/1e9 for r in measured),
            'host_RSS_max_observed_gb':max(r['process_memory']['rss_current_bytes']/1e9 for r in rows),
            'host_RSS_process_highwater_gb':max(r['process_memory']['rss_process_highwater_bytes']/1e9 for r in rows),
            'close':data.get('close'),'results_sha256':sha((folder/'results.json').read_bytes())})
        all_rows.extend(dict(r,job=job['name']) for r in rows)
    comparisons=[];stable=[]
    for n in protocol['input_lengths']:
        dt=next(c for c in cells if c['method']=='deltatrace_streamed' and c['input_length']==n)
        for method in ['ifr_multi_hop_both','ifr_multi_hop']:
            ft=next(c for c in cells if c['method']==method and c['input_length']==n)
            d={'input_length':n,'FT_method':method,'actual_total_tokens':dt['actual_total_tokens']}
            for key in ['allocated','reserved']:
                a=dt['peak_'+key+'_gb'];b=ft['peak_'+key+'_gb']
                d.update({f'DT_peak_{key}_gb':a,f'FT_peak_{key}_gb':b,f'{key}_reduction_percent':100*(1-a/b),f'{key}_passed':a<b})
            d['acceptance_passed']=d['allocated_passed'] and d['reserved_passed'];comparisons.append(d)
            old_dt=[c for c in speed['cost_cells'] if c['method']=='deltatrace_streamed' and c['target_input_tokens']==n]
            old_ft=[c for c in speed['cost_cells'] if c['method']==method and c['target_input_tokens']==n]
            check={'input_length':n,'FT_method':method}
            for key in ['allocated','reserved']:
                a=max(c['steady_peak_'+key+'_gb'] for c in old_dt);b=max(c['steady_peak_'+key+'_gb'] for c in old_ft)
                check.update({f'DT_steady_{key}_gb':a,f'FT_steady_{key}_gb':b,f'{key}_passed':a<b})
            check['acceptance_passed']=check['allocated_passed'] and check['reserved_passed'];stable.append(check)
    scoped_passed=speed['speed_acceptance_passed'] and all(c['acceptance_passed'] for c in comparisons+stable)
    report={'status':'verified','protocol_sha256':sha(protocol_path.read_bytes()),'cells':cells,'independent_geometry_memory_comparisons':comparisons,
        'original_two_round_warm_memory_comparisons':stable,'independent_cold_and_warm_memory_passed':all(c['acceptance_passed'] for c in comparisons),
        'original_two_round_warm_speed_passed':speed['speed_acceptance_passed'],'fixed_geometry_speed_and_memory_passed':scoped_passed,
        'mixed_geometry_strict_per_bin_cold_transition_gate_passed':False,
        'transition_limitation':'Descending905-to478 and478-to265 sequences enter with the previous larger private graph pool resident. Its physical bytes count in that smaller-bin transition peak, which exceeds the corresponding FT peak. The failed frozen gate and every original row remain; no peak reset after old-pool release and no row exclusion is used to hide it.',
        'scope':'Original two-round warm complete-API speed with95% bootstrap; original per-length stable allocated/reserved memory; separate fresh-process per-length capacity including every cold build. This is not a claim of faster graph creation or uniformly lower per-bin cold transition memory for arbitrary shape histories.',
        'new_speed_observations_used_for_acceptance':False,'isolated_full_API_calls':{'timed':60,'audit':12,'retained_reference':4,'total':76},
        'setup_failure':'Original queue wrongly required an unused Triton-cache directory for eager FT before starting FT. Zero FT calls occurred; corrected controller resumed11 cells, retained first DT128 unchanged. queue_setup_failure.json preserves original state.',
        'all_input_hashes_matched':True,'all_DT_full_vectors_and_math_exact':True,'excluded_rows':0,
        'cost_units':'Decimal GB. Host RSS current is observed after calls; process highwater is lifetime, not a per-call host peak. Warm graph allocated does not represent its retained temporary pool; physical reserved and cold allocated are both required.'}
    with (HERE/'isolated_memory_all_rows.csv').open('w',newline='') as f:
        flat=[{k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in all_rows]
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);a=p.parse_args();report=summarize(a.raw)
    (HERE/'isolated_memory_summary.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['status','fixed_geometry_speed_and_memory_passed','independent_cold_and_warm_memory_passed','mixed_geometry_strict_per_bin_cold_transition_gate_passed','independent_geometry_memory_comparisons']}))
