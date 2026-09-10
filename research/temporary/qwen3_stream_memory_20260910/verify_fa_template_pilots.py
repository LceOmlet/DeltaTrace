"""Verify complete long-bin pilots independently of the GPU driver assertions."""
import json
from verify_target_evidence import S,archive,cases,sha
from verify_fa_owner_evidence import verify as operators

def verify():
    op=operators();result=[]
    for phase,owner in [('fa_template','fa_owner64_warps4'),('fa_cached_template','fa_owner_cached')]:
        files,receipt=archive(phase);r=cases(files,phase,phase,'input',[1024])[0];p=json.loads(files[phase+'_protocol.json']);d=json.loads(files[phase+'/results.json']);c=d['cases'][0]
        reference=next(x for x in op['candidates'] if x['phase']==owner)
        assert p['experimental_finite_library']['sha256']==reference['library_sha256']
        assert d['experimental_finite_library']==p['experimental_finite_library']
        assert all(c['metadata_mutation_control'][k] for k in ['vector_exact','math_exact','fresh_nested_metadata'])
        assert c['details']['output_materialization']['fresh_mutable_containers_each_return'] and not c['details']['output_materialization']['attribution_values_reused']
        r.update(archive_sha256=receipt['sha256'],finite_library_sha256=reference['library_sha256'],metadata_mutation_control=c['metadata_mutation_control'],full_API_calls=8)
        result.append(r)
    return {'status':'verified','pilots':result,'full_API_calls':16,'operators':op,'formal_FT_confirmation_complete':False,
        'scope':'Complete long-bin pilot gains and strict values verified. No final acceptance before separate production-source confirmation and original-case validation.'}

if __name__=='__main__':
    d=verify();(S/'fa_template_pilots_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps({k:v for k,v in d.items() if k!='operators'},indent=2))
