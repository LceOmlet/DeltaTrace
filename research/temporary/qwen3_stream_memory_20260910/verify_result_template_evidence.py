"""Verify the insufficient metadata-only optimization and its GPU profile."""
import json,statistics
from verify_target_evidence import S,archive,cases

def verify():
    result=[]
    for phase in ['result_template','result_profile']:
        files,receipt=archive(phase);record=cases(files,phase,phase,'input',[1024])[0]
        d=json.loads(files[phase+'/results.json']);c=d['cases'][0];control=c['metadata_mutation_control']
        assert all(control[k] for k in ['vector_exact','math_exact','fresh_nested_metadata'])
        meta=c['details']['output_materialization'];assert meta['fresh_mutable_containers_each_return'] and not meta['attribution_values_reused']
        record.update(archive_sha256=receipt['sha256'],metadata=meta,caller_metadata_mutation_control=control,
                      full_API_calls=9 if phase=='result_profile' else 8)
        if phase=='result_profile':
            assert c['profile']['vector_exact'] and c['profile']['math_exact']
            trace=json.loads(files[phase+'/trace.json']);kernels=[r for r in trace['traceEvents'] if r.get('cat')=='kernel' and 'dur' in r]
            assert kernels
            cats={'finite_FA':lambda n:'deltatrace_fa_finite_p1_kernel' in n,'native_FA':lambda n:'flash_fwd_kernel<' in n,
                  'native_MM':lambda n:'mcblas__Mck_hgemm' in n}
            record['profile']={'complete_API_seconds':c['profile']['seconds'],'host_sections':c['profile']['host_sections'],
                'kernel_count':len(kernels),'sum_kernel_milliseconds':sum(r['dur'] for r in kernels)/1000,
                'categories':{key:{'calls':sum(fn(r['name']) for r in kernels),'milliseconds':sum(r['dur'] for r in kernels if fn(r['name']))/1000} for key,fn in cats.items()},
                'scope':'Separate instrumented call; its timing is not a replacement for any measured latency row.'}
        result.append(record)
    return {'status':'verified','candidates':result,'production_runtime_changed':False,
            'selected_as_final_runtime':False,'reason':'Mean about540ms remains above v2 FT plain long-bin533.8–539.4ms; metadata-only gain is insufficient.'}

if __name__=='__main__':
    d=verify();(S/'result_template_evidence_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
