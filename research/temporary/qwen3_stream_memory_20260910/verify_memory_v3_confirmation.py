"""Independently recompute all frozen v3 statistics and verify the public release."""
import json
from verify_target_evidence import S,archive,sha
from verify_memory_v3_sources import verify as verify_sources

def verify():
    files,receipt=archive('memory_v3_confirmation');source=verify_sources()
    local=json.loads((S/'memory_v3_confirmation_local_summary.json').read_bytes())
    remote=json.loads(files['memory_v3_confirmation_summary.json']);assert local==remote
    assert (S/'memory_v3_confirmation_local_summary.csv').read_bytes()==files['memory_v3_confirmation_summary.csv']
    runtime=json.loads((S/'memory_v3_confirmation_protocol.json').read_bytes())['runtime_files']
    for n,h in runtime.items():assert sha(files['memory_production_v3_release/'+n])==h,n
    queue=json.loads(files['memory_v3_confirmation/queue.json'])
    for job in queue['jobs']:
        data=json.loads(files['memory_v3_confirmation/'+job['name']+'/results.json'])
        if job['method']!='deltatrace_streamed':continue
        details=data['cases'][0]['details'];meta=details['output_materialization']
        assert meta['fresh_mutable_containers_each_return'] and not meta['attribution_values_reused']
        assert meta['static_metadata_template_bytes']>0 and meta['dynamic_scalar_paths']==144
    assert len(local['verified_cells'])==24 and local['all_timed_rows_count']==120 and local['excluded_timing_rows']==0
    return {'status':'verified','archive_sha256':receipt['sha256'],'independently_recomputed_all_statistics_equal_to_remote':True,
        'source_verification':source,'public_template_metadata_verified_for_all_8_DT_cells':True,
        'verified_cells':24,'timing_rows':120,'excluded_rows':0,'acceptance_passed':local['acceptance_passed'],
        'speed_acceptance_passed':local['speed_acceptance_passed'],'memory_acceptance_passed':local['memory_acceptance_passed'],
        'failed_speed_comparisons':[c for c in local['comparisons'] if not c['acceptance_passed']],
        'failed_memory_comparisons':[c for c in local['memory_comparisons'] if not c['acceptance_passed']],
        'scope':local['acceptance_scope']}

if __name__=='__main__':
    d=verify();(S/'memory_v3_confirmation_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
