"""Verify the final all-file audit against its predeclared identity plan."""
import json
from verify_target_evidence import S,archive,sha

def verify():
    files,receipt=archive('memory_v4_postflight');p=json.loads(files['postflight_memory_v4_plan.json']);d=json.loads(files['postflight_memory_v4.json'])
    assert d['status']=='verified' and d['model_calls']==d['attribution_calls']==d['GPU_program_calls']==0
    expected={(r['path'],r['kind']):r['sha256'] for r in p['files']}
    for n,h in p['normalized_FT'].items():expected[('/root/flashtrace-vjp-official/'+n,'fixed_FT_normalized_source')]=h
    for r in d['files']:
        assert r['matches'] and r['actual_sha256']==r['sha256']
        key=(r['path'],r['kind'])
        if key in expected:assert r['actual_sha256']==expected.pop(key)
        elif r['kind']=='original_native_model_source':assert r['actual_sha256']==p['native_model_sha256']
        elif r['kind']=='original_public_SAC_source':assert r['actual_sha256']==p['native_SAC_sha256']
        else:assert r['kind']=='unchanged_original_finite_FA_binary'
    assert not expected and len(d['files'])==211
    assert 'no process found' in d['GPU_status_text'].lower() and files['final_memory_v4_mx_smi.txt'].decode()==d['GPU_status_text']
    followup=json.loads(files['memory_v4_followup.json']);assert followup['status']=='complete' and len(followup['jobs'])==2
    return {'status':'verified','archive_sha256':receipt['sha256'],'identity_checks':len(d['files']),'all_original_checkpoint_model_FA_FT_and_SAC_sources_preserved':True,'current_v4_public_sources_and_common_loader_identical':True,'final_GPU_has_no_processes':True,'model_calls':0,'attribution_calls':0}

if __name__=='__main__':
    d=verify();(S/'memory_v4_postflight_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
