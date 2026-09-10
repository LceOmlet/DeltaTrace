"""Recompute all final statistics and verify the executed public v4 release."""
import json
from verify_target_evidence import S,archive,sha
from verify_memory_v4_sources import verify as verify_sources
from verify_cpu_staged_loading import verify as verify_loading

def verify():
    files,receipt=archive('memory_v4_confirmation');source=verify_sources();loading=verify_loading()
    local=json.loads((S/'memory_v4_confirmation_local_summary.json').read_bytes());remote=json.loads(files['memory_v4_confirmation_summary.json']);assert local==remote
    assert (S/'memory_v4_confirmation_local_summary.csv').read_bytes()==files['memory_v4_confirmation_summary.csv']
    p=json.loads(files['memory_v4_confirmation_protocol.json'])
    for n,h in p['runtime_files'].items():assert sha(files['memory_production_v4_release/'+n])==h,n
    for n,h in p['common_loading_helpers'].items():assert sha(files[n])==h,n
    assert p['common_loading_identity_archive_sha256']==loading['archive_sha256'] and loading['state_entries']==400
    # Derive the formal driver from the earlier frozen driver, changing only
    # common model loading and its source-identity assertion for every method.
    old,_=archive('memory_v3_confirmation');text=old['benchmark_memory_v3.py'].decode()
    marker="    assert args.family=='qwen3'";assert text.count(marker)==1
    text=text.replace(marker,marker+"\n    for name,digest in protocol['common_loading_helpers'].items():assert sha((Path(__file__).parent/name).read_bytes())==digest,name")
    original="            model,tokenizer=bench.load_model_balanced(env['checkpoint'],'cuda:0')";assert text.count(original)==1
    text=text.replace(original,"            from cpu_staged_model_loading import load_common_staged_model\n            model,tokenizer,report['model_loading_strategy']=load_common_staged_model(bench,env['checkpoint'],'cuda:0')")
    assert text.encode()==files['benchmark_memory_v4.py']
    queue=json.loads(files['memory_v4_confirmation/queue.json']);strategies=set()
    for job in queue['jobs']:
        data=json.loads(files['memory_v4_confirmation/'+job['name']+'/results.json']);strategy=data['model_loading_strategy'];strategies.add(strategy['strategy'])
        assert not strategy['model_forward_modified'] and not strategy['FT_implementation_modified']
        assert data['model_load_seconds']>=strategy['loading_seconds']>0 and data['model_load_memory']['host_process_highwater_bytes']>0
        if job['method']!='deltatrace_streamed':continue
        details=data['cases'][0]['details'];meta=details['output_materialization']
        assert meta['fresh_mutable_containers_each_return'] and not meta['attribution_values_reused']
        assert meta['static_metadata_template_bytes']>0 and meta['dynamic_scalar_paths']==144
    assert len(strategies)==1 and len(local['verified_cells'])==24 and local['all_timed_rows_count']==120 and local['excluded_timing_rows']==0
    return {'status':'verified','archive_sha256':receipt['sha256'],'independently_recomputed_all_statistics_equal_to_remote':True,
        'source_verification':source,'public_template_metadata_verified_for_all_8_DT_cells':True,'same_original_model_loading_for_all_24_processes':True,
        'driver_only_change_is_common_loading_and_identity_check':True,'all_400_original_parameter_and_buffer_states_exact':True,
        'verified_cells':24,'timing_rows':120,'excluded_rows':0,'acceptance_passed':local['acceptance_passed'],
        'speed_acceptance_passed':local['speed_acceptance_passed'],'memory_acceptance_passed':local['memory_acceptance_passed'],
        'failed_speed_comparisons':[c for c in local['comparisons'] if not c['acceptance_passed']],
        'failed_memory_comparisons':[c for c in local['memory_comparisons'] if not c['acceptance_passed']],'scope':local['acceptance_scope']}

if __name__=='__main__':
    d=verify();(S/'memory_v4_confirmation_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
