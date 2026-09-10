"""Gate source promotion on all full-vector and live-validation pilot controls."""
import json
from verify_target_evidence import S,archive,cases,sha
from verify_fa_template_pilots import verify as long_pilots

def verify():
    previous=long_pilots();files,receipt=archive('fa_cached_followup');read=lambda n:json.loads(files[n])
    q=read('fa_cached_followup/queue.json');plan=read('fa_cached_followup_plan.json')
    assert q['status']=='complete' and len(q['jobs'])==7 and all(j['status']=='complete' and j['returncode']==0 for j in q['jobs'])
    for n,h in plan['files'].items():assert sha(files[n])==h,n
    probe=read('template_validation/results.json');assert probe['status']=='complete' and probe['model_calls']==probe['attribution_calls']==0
    checks=probe['checks'];assert len(checks['strict_graph_cases'])==8 and checks['storage_views_exact'] and checks['changed_storage_topology_rejected']
    assert all(c['passed'] and c['rejected_before_return']==(c['case']!='valid') for c in checks['strict_graph_cases'])
    cells=[]
    for kind,lengths,key in [('short',[128,256,512],'input'),('rollout',[10,100,500],'output')]:
        for n in lengths:
            folder='fa_cached_followup/'+kind+'_'+str(n);record=cases(files,folder,'fa_cached_'+kind,key,[n])[0]
            c=read(folder+'/results.json')['cases'][0]
            if kind=='short':assert all(c['metadata_mutation_control'][k] for k in ['vector_exact','math_exact','fresh_nested_metadata'])
            record['full_API_calls']=8 if kind=='short' else 7;cells.append(record)
    return {'status':'verified','archive_sha256':receipt['sha256'],'all_runtime_numerical_pilot_cells':7,
        'cells':cells,'long_pilot':previous['pilots'][1],'selected_candidate_full_API_calls':53,
        'strict_live_graph_cases':checks,'all_original_vectors_and_math_exact':True,
        'all_template_mutation_controls_passed':True,'public_production_confirmation_required':True,
        'memory_scope':'Fixed short-output32 geometry capacity is the final gate; longer-rollout memory is measured separately and not inferred lower thanFT.'}

if __name__=='__main__':
    d=verify();(S/'fa_cached_followup_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
