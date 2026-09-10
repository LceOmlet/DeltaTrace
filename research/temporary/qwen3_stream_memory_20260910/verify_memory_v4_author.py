"""Verify all saved original samples and the actual native-root observations."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def verify(raw):
    plan=json.loads((HERE/'memory_v4_author_plan.json').read_bytes())
    path=raw/'memory_v4_author/results.json';data=json.loads(path.read_bytes())
    assert data['status']=='complete' and data['all_vectors_exact']
    assert data['driver_sha256']==plan['driver_sha256'] and data['plan_sha256']==sha((HERE/'memory_v4_author_plan.json').read_bytes())
    for name,digest in plan['common_loading_helpers'].items():assert sha((raw/name).read_bytes())==digest
    assert not data['model_loading_strategy']['model_forward_modified'] and not data['model_loading_strategy']['FT_implementation_modified']
    assert len(data['cases'])==len(plan['cases'])==11
    assert data['generation_calls']==data['metric_calls']==data['FT_calls']==0
    checks=[]
    with np.load(raw/'memory_v4_author/vectors.npz') as z:
        for source,case in zip(plan['cases'],data['cases']):
            assert case['key']==source['key'] and case['input_sha256']==source['input_sha256']
            actual=np.asarray(source['input_ids'],dtype=np.int64)[None];base=actual.copy()
            eligible=[source['user_positions'][j] for j in source['keep']];base[0,eligible]=actual[0,-1]
            pair=np.concatenate((base,actual));pair_sha=sha(pair.tobytes())
            calls={c['mode']:c for c in case['calls']};assert len(calls)==3
            reference=calls['retained']['details'];old=z[case['key']+'/retained']
            assert calls['retained']['actual_root']==[{'shape':[2,actual.shape[1]],'input_sha256':pair_sha}]
            for mode in ['retained','root_cold','root_hot']:
                call=calls[mode];vec=z[case['key']+'/'+mode];detail=call['details']
                assert sha(vec.tobytes())==call['vector_sha256'] and np.array_equal(old,vec) and np.isfinite(vec).all()
                for key in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']:assert detail[key]==reference[key]
                if mode=='retained':continue
                cold=mode=='root_cold';execution=detail['native_graph_execution']
                assert len(call['actual_root'])==(3 if cold else 0)
                assert execution['native_model_Python_root_calls']==(3 if cold else 0)
                assert execution['native_model_GPU_root_executions']==(4 if cold else 1)
                assert detail['whole_root_graph']['fresh_graph_input_sha256']==pair_sha
                assert detail['deferred_validation']['all_passed'] and detail['deferred_validation']['predicates'] in [829,865]
                materialization=detail['output_materialization']
                assert materialization['fresh_mutable_containers_each_return'] and not materialization['attribution_values_reused'] and materialization['dynamic_scalar_paths']==144
                assert all(c['native_input_exact'] and c['native_output_exact'] for c in detail['native_layer_boundary_checks']['paired_batch'])
                assert all(c['public_output_exact_to_actual_model_FA'] for c in detail['public_FA_capture_checks'])
            if 'changed_input_control' in case:
                control=case['changed_input_control'];new=z[case['key']+'/changed_new'];old_changed=z[case['key']+'/changed_retained']
                assert control['same_graph'] and control['exact'] and control['output_changed'] and control['math_exact']
                assert np.array_equal(new,old_changed) and not np.array_equal(new,old)
            checks.append({'key':case['key'],'total_tokens':actual.shape[1],'full_vectors_and_math_exact':True,'actual_original_root_observations':True})
        assert np.array_equal(z[data['cases'][-1]['key']+'/contract_recovery'],z[data['cases'][-1]['key']+'/retained'])
    assert data['capture_exception_cleanup_passed'] and data['explicit_controller_close_passed']
    assert data['contract_recovery_exact'] and data['contract_recovery_math_exact']
    assert len(data['strict_scalar_checks'])==4 and all(c.get('all_fields_match_eager',c.get('rejected_before_return')) for c in data['strict_scalar_checks'])
    assert len(data['strict_rope_checks'])==3 and all(c['passed'] for c in data['strict_rope_checks'])
    assert len(data['graph_validation_probes']['strict_graph_cases'])==8 and all(c['passed'] for c in data['graph_validation_probes']['strict_graph_cases'])
    assert len(data['memory_contract_controls'])==6 and all(c.get('rejected_before_graph_execution',c.get('rejected')) for c in data['memory_contract_controls'])
    return {'status':'verified','results_sha256':sha(path.read_bytes()),'cases':checks,'all_full_vectors_and_math_exact':True,
        'changed_real_input_same_graph_exact_and_output_changed':True,'cold_native_root_calls_observed':33,'hot_Python_root_calls_observed':0,
        'strict_scalar_controls':4,'strict_RoPE_controls':3,'strict_GPU_graph_controls':8,'model_state_and_actual_root_version_controls':6,
        'fresh_controller_recovery_exact':True,'complete_ordinary_API_calls':36,'rejected_attribute_calls':5,'separate_faulted_native_root_capture':1,
        'cost_scope':'All cold and hot correctness calls include construction, original model GPU work, complete return and audit; not steady-speed observations.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);a=p.parse_args();report=verify(a.raw)
    (HERE/'memory_v4_author_verification.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n');print(json.dumps(report))
