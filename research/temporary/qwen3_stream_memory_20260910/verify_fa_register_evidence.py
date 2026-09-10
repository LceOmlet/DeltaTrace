"""Independently verify all retained register-reuse operator and compile trials."""
import json
from verify_target_evidence import S,archive,sha

def verify():
    records=[];sources={}
    for phase in ['fa_all_cached','fa_all64_cached','fa_mma_owner','fa_mma_owner_v2']:
        f,receipt=archive(phase);read=lambda n:json.loads(f[phase+'/'+n])
        p=read('protocol.json');d=read('results.json');source=f[phase+'/vendor_fa_finite_p1_shared_mean_reuse.cu'];sources[phase]=source
        assert sha(source)==p['extension_sha256'] and d['protocol']==p
        assert d['vendor_sources_before']==d['vendor_sources_after'] and len(d['vendor_sources_before'])==78
        assert all(d[k]==0 for k in ['native_model_forwards','native_model_backwards','attribution_calls','GPU_kernel_launches'])
        row={'phase':phase,'archive_sha256':receipt['sha256'],'source_sha256':sha(source),'operator_calls':0}
        if phase=='fa_mma_owner':
            assert d['status']=='finite_extension_compile_failed' and phase+'/operator_results.json' not in f
        else:
            assert d['compile_returncode']==0 and sha(f[phase+'/libdeltatrace_fa_finite_shared_mean_reuse.so'])==d['library']['sha256']
            op=read('operator_results.json');protocol=read('operator_protocol.json')
            assert op['status']=='complete' and op['operator_calls']==100
            assert op['protocol_sha256']==sha(f[phase+'/operator_protocol.json']) and op['driver_sha256']==sha(f[phase+'/finite_operator.py'])
            assert all(op[k]==0 for k in ['model_calls','backward_calls','full_attribution_calls'])
            assert {(c['batch'],c['length']) for c in op['cases']}=={(b,n) for b in [1,4] for n in [32,33,101,905,1031]}
            assert len(op['cases'])==10
            for c in op['cases']:
                assert c['all_exact'] and len(c['rows'])==10 and set(c['comparisons'])=={'dq','dk','dv','tau','center'}
                for v in c['comparisons'].values():assert v['exact'] and v['max_abs']==0 and v['old_sha256']==v['new_sha256']
            row.update(operator_calls=100,all_five_fields_bit_exact=True,library_sha256=d['library']['sha256'])
        records.append(row)
    assert sources['fa_mma_owner'].count(b'std::max(size(')==2
    assert sources['fa_mma_owner'].replace(b'std::max(size(',b'std::max<int>(size(')==sources['fa_mma_owner_v2']
    assert b'flash::gemm<true, false>' in sources['fa_mma_owner_v2'] or b'flash::gemm<true,false>' in sources['fa_mma_owner_v2']
    return {'status':'verified','trials':records,'operator_calls':300,'model_calls':0,'full_attribution_calls':0,'all_operator_outputs_bit_exact':True,'failed_compile_preserved':True,'repair_only_two_host_integer_template_arguments':True,'original_vendor_sources_unchanged':True}

if __name__=='__main__':
    d=verify();(S/'fa_register_evidence_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
