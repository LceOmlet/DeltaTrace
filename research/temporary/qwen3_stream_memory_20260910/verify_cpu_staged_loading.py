"""Verify all model-state fingerprints and both complete loading-cost records."""
import json
from verify_target_evidence import S,archive,sha

def verify():
    f,r=archive('cpu_staged_loading');p=json.loads(f['cpu_staged_loading_plan.json']);d=json.loads(f['cpu_staged_loading/results.json'])
    assert d['status']=='complete' and d['model_loads']==2 and d['model_forwards']==d['model_backwards']==d['attribution_calls']==0
    assert d['driver_sha256']==p['driver_sha256']==sha(f['check_cpu_staged_loading.py']) and d['plan_sha256']==sha(f['cpu_staged_loading_plan.json'])
    old,new=d['records'];assert old['mode']=='original_GPU_load' and new['mode']=='original_CPU_load_then_native_to_GPU'
    assert old['state']==new['state'] and old['tokenizer_special_ids']==new['tokenizer_special_ids']
    assert all(v['device']=='cuda:0' for v in new['state'].values())
    assert old['after_release_allocated_gb']==new['after_release_allocated_gb']==0
    assert new['peak_allocated_gb']<old['peak_allocated_gb'] and new['peak_reserved_gb']<old['peak_reserved_gb']
    assert old['all_parameters_on_C550'] and new['all_parameters_on_C550']
    return {'status':'verified','archive_sha256':r['sha256'],'all_parameter_and_buffer_state_bytes_exact':True,'state_entries':len(new['state']),
        'records':[{k:v for k,v in c.items() if k!='state'} for c in d['records']],
        'applies_to':'Common original author CPU loader followed by unmodified model.to(cuda:0), used identically for DT and both FT. No model/FT computation edited; full loading GPU and host memory costs retained.',
        'timing_limitation':'One loader pair in this process, not a stable loading-speed estimate. Final independent processes record their own complete load cost.',
        'model_loads':2,'attribution_calls':0,'model_calls_for_this_verifier':0}

if __name__=='__main__':
    d=verify();(S/'cpu_staged_loading_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
