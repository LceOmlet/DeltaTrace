"""Compare independently recomputed local v2 statistics with frozen GPU receipt."""
import json
from verify_target_evidence import S,archive,sha

def verify():
    files,receipt=archive('memory_v2_confirmation')
    local=json.loads((S/'memory_v2_confirmation_local_summary.json').read_bytes());remote=json.loads(files['memory_v2_confirmation_summary.json'])
    assert local==remote
    assert (S/'memory_v2_confirmation_local_summary.csv').read_bytes()==files['memory_v2_confirmation_summary.csv']
    assert not local['acceptance_passed'] and not local['speed_acceptance_passed']
    return {'status':'verified','archive_sha256':receipt['sha256'],'independently_recomputed_all_statistics_equal_to_remote':True,
        'verified_cells':len(local['verified_cells']),'timing_rows':local['all_timed_rows_count'],'excluded_rows':local['excluded_timing_rows'],
        'acceptance_passed':False,'failed_speed_comparisons':[c for c in local['comparisons'] if not c['acceptance_passed']],
        'scope':'This frozen v2 experiment failed; later candidates must have new sources and new confirmation records.'}

if __name__=='__main__':
    d=verify();(S/'memory_v2_confirmation_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
