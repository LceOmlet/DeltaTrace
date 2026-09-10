"""Check raw full-call pilot arrays, unchanged math, and live current returns."""
import io,json,statistics
import numpy as np
from verify_target_evidence import S,archive,sha,KEYS
from verify_fa_register_evidence import verify as operators

def verify():
    op=operators();f,receipt=archive('fa_mma_pilot');v3,_=archive('memory_v3_confirmation');q=json.loads(f['fa_mma_pilot/queue.json'])
    assert q['status']=='complete' and len(q['jobs'])==2 and all(j['status']=='complete' and j['returncode']==0 for j in q['jobs'])
    cells=[]
    for kind,n in [('short',1024),('rollout',100)]:
        prefix='fa_mma_pilot_'+kind;p=json.loads(f[prefix+'_protocol.json']);folder='fa_mma_pilot/'+kind;d=json.loads(f[folder+'/results.json'])
        assert d['status']=='complete' and d['driver_sha256']==p['driver_sha256']==sha(f['benchmark_'+prefix+'.py'])
        assert d['protocol_sha256']==sha(f[prefix+'_protocol.json'])
        for name,h in p['pilot_helpers'].items():assert sha(f[name])==h
        for name,h in p['runtime_files'].items():assert sha(v3['memory_production_v3_release/'+name])==h
        assert p['experimental_finite_library']['sha256']==op['trials'][-1]['library_sha256']
        assert len(d['cases'])==1 and len(d['rows'])==5 and all(r['status']=='ok' for r in d['rows'])
        c=d['cases'][0]
        with np.load(io.BytesIO(f[folder+'/vectors.npz'])) as z:
            vec=z[str(n)];old=z[str(n)+'_retained'];assert np.array_equal(vec,old) and np.isfinite(vec).all()
            assert sha(vec.tobytes())==c['audit']['vector_sha256']
        assert all(c['details'][k]==c['retained_details'][k] for k in KEYS)
        assert c['details']['deferred_validation']['all_passed']
        meta=c['details']['output_materialization'];assert meta['fresh_mutable_containers_each_return'] and not meta['attribution_values_reused'] and meta['dynamic_scalar_paths']==144
        if kind=='short':assert all(c['metadata_mutation_control'][k] for k in ['vector_exact','math_exact','fresh_nested_metadata'])
        measured=[r for r in d['rows'] if r['phase']=='measured'];assert len(measured)==3
        cells.append({'kind':kind,'length':n,'full_API_calls':8 if kind=='short' else 7,'mean_seconds':statistics.mean(r['time_sec'] for r in measured),'model_load_memory':d['model_load_memory'],'all_call_peak_allocated_gb':max(r['peak_allocated_gb'] for r in d['rows']),'all_call_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in d['rows']),'original_retained_full_vector_and_math_exact':True})
    return {'status':'verified','archive_sha256':receipt['sha256'],'operator_evidence':op,'cells':cells,'full_API_calls':15,'all_original_vectors_and_math_exact':True,'fresh_current_return_verified':True,'final_acceptance_requires_new_public_runtime_confirmation':True}

if __name__=='__main__':
    d=verify();(S/'fa_mma_pilot_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
