"""Verify all preselected original short samples, complete vectors and strict controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def verify(raw,study):
    sha=lambda b:hashlib.sha256(b).hexdigest();plan=json.loads((study/'author_graph_plan.json').read_bytes())
    data=json.loads((raw/'author_graph/results.json').read_bytes());post=json.loads((raw/'postflight.json').read_bytes())
    assert data['status']=='complete' and data['driver_sha256']==plan['driver_sha256']
    assert data['plan_sha256']==sha((study/'author_graph_plan.json').read_bytes())
    assert data['all_vectors_exact'] and data['capture_exception_cleanup_passed'] and data['explicit_controller_close_passed']
    assert data['generation_calls']==data['metric_calls']==data['FT_calls']==0 and data['sample_batch']==1
    assert [c['key'] for c in data['cases']]==[c['key'] for c in plan['cases']] and len(plan['cases'])==11
    assert len(data['strict_scalar_checks'])==4 and len(data['strict_rope_checks'])==3
    probes=data['graph_validation_probes'];assert len(probes['strict_graph_cases'])==8
    assert all(c['passed'] and c['rejected_before_return']==(c['case']!='valid') for c in probes['strict_graph_cases'])
    assert probes['storage_views_exact'] and probes['changed_storage_topology_rejected']
    assert probes['strict_graph_replays']==8 and probes['model_calls']==probes['FA_calls']==probes['attribution_calls']==0
    rows=[];calls=0
    with np.load(raw/'author_graph/vectors.npz') as vectors:
        assert len(vectors.files)==22
        for index,(given,result) in enumerate(zip(plan['cases'],data['cases'])):
            ids=np.asarray(given['input_ids'],dtype=np.int64)
            assert sha(ids.tobytes())==given['input_sha256']==result['input_sha256'] and len(ids)==result['total_tokens']<=1024
            assert result['exact'] and result['all_math_diagnostics_exact'] and len(result['calls'])==2
            assert [c['mode'] for c in result['calls']]==(['retained','root'] if index%2==0 else ['root','retained'])
            assert np.array_equal(vectors[given['key']+'/root'],vectors[given['key']+'/retained'])
            for call in result['calls']:
                vec=vectors[given['key']+'/'+call['mode']];assert np.isfinite(vec).all() and vec.shape==(len(ids),)
                assert sha(vec.tobytes())==call['vector_sha256'];calls+=1
                assert len(call['actual_root'])==1 and call['actual_root'][0]['shape']==[2,len(ids)]
                if call['mode']=='root':
                    d=call['details'];g=d['native_graph_execution']
                    assert d['deferred_validation']['all_passed'] and d['deferred_validation']['predicates']==829
                    assert d['root_retention_mutation_audit']=={'enabled':True,'predicates':720}
                    assert d['graph_input_copy_audit']=={'enabled':True,'predicates':883,'all_passed':True}
                    assert g['every_input_tensor_refreshed'] and not g['attribution_results_reused']
                    assert g['geometry_cache_entries']==1 and g['graph_replays']==1 and g['native_model_root_calls']==1
                    assert g['finite_program_warmups_this_call']==2 and g['build_this_call']['native_graph_program_recordings']==1
                    assert d['native_layer_replay_calls']==0 and d['extra_native_fa_attention_calls']==36
                    assert all(c['public_output_exact_to_actual_model_FA'] for c in d['public_FA_capture_checks'])
                    assert all(c['native_input_exact'] and c['native_output_exact'] for c in d['native_layer_boundary_checks']['paired_root'])
            first,second=result['calls'];assert first['actual_root']==second['actual_root']
            for key in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']:assert first['details'][key]==second['details'][key]
            rows.append({'key':given['key'],'complete_length':len(ids),'exact_complete_vector':True,'exact_all_math_diagnostics':True,
                'seconds_including_build_and_audit':{c['mode']:c['seconds_including_compilation_and_audit'] for c in result['calls']},
                'peak_allocated_gb':{c['mode']:c['reported_peak_allocated_gb'] for c in result['calls']}})
    assert post['status']=='verified' and len(post['files'])==111 and all(r['matches'] for r in post['files'])
    assert 'no process found' in post['GPU_status_text']
    return {'verified_cases':len(rows),'complete_attribution_calls':calls,'all_full_vectors_and_math_exact':True,
        'cases':rows,'separate_validation_seconds':data['validation_probe_seconds']+probes['seconds'],
        'strict_graph_probe_cases':probes['strict_graph_cases'],'capture_exception_cleanup_passed':True,'explicit_controller_close_passed':True,
        'source_and_checkpoint_postflight_files_verified':len(post['files']),'GPU_idle_after_completion':True,
        'new_generation_calls':0,'new_metric_calls':0,'author_plan_sha256':sha((study/'author_graph_plan.json').read_bytes()),
        'raw_results_sha256':sha((raw/'author_graph/results.json').read_bytes()),'raw_vectors_sha256':sha((raw/'author_graph/vectors.npz').read_bytes()),
        'postflight_sha256':sha((raw/'postflight.json').read_bytes()),
        'scope':'All11 original complete<=1k development cases, no truncation or outcome selection. Audit/build calls establish compatibility, not independent warm performance.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.raw,a.study);a.output.write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','strict_graph_probe_cases']},indent=2))
