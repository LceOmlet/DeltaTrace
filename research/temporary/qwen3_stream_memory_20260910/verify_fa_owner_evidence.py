"""Verify both failed owner-row candidates and the bounded scheduling repair."""
import io,json,statistics
import numpy as np
from verify_target_evidence import S,archive,sha

def verify():
    summary=[]
    for phase in ['fa_owner64','fa_owner64_split','fa_owner64_warps4','fa_owner_cached']:
        files,receipt=archive(phase);read=lambda n:json.loads(files[phase+'/'+n]);p=read('protocol.json');op=read('operator_protocol.json');build=read('results.json');d=read('operator_results.json')
        assert build['status']=='finite_extension_compiled_not_executed' and build['vendor_sources_before']==build['vendor_sources_after']
        assert build['protocol']==p and p['extension_sha256']==sha(files[phase+'/vendor_fa_finite_p1_shared_mean_reuse.cu'])
        assert build['library']['sha256']==sha(files[phase+'/libdeltatrace_fa_finite_shared_mean_reuse.so'])
        assert d['driver_sha256']==op['driver_sha256']==sha(files[phase+'/finite_operator.py']) and d['protocol_sha256']==sha(files[phase+'/operator_protocol.json'])
        assert d['model_calls']==d['backward_calls']==d['full_attribution_calls']==0
        assert d['operator_calls']==sum(len(c['rows']) for c in d['cases'])
        cells=[]
        for c in d['cases']:
            assert len(c['rows'])==10 and len(c['comparisons'])==5
            assert c['all_exact']==all(x['exact'] for x in c['comparisons'].values())
            for x in c['comparisons'].values():assert x['exact']==(x['old_sha256']==x['new_sha256'])
            times={m:[r['seconds'] for r in c['rows'] if r['method']==m and r['phase']=='measured'] for m in ['old','new']};assert all(len(v)==3 for v in times.values())
            cells.append({'batch':c['batch'],'length':c['length'],'all_outputs_exact':c['all_exact'],'old_mean_seconds':statistics.mean(times['old']),'new_mean_seconds':statistics.mean(times['new']),'new_to_old_ratio':statistics.mean(times['new'])/statistics.mean(times['old'])})
        if phase=='fa_owner64':
            assert d['status']=='failed' and d['operator_calls']==10
            with np.load(io.BytesIO(files[phase+'/failed_1_32.npz'])) as v:
                for k in ['dq','dk','dv','tau','center']:
                    exact=bool(np.array_equal(v['old_'+k],v['new_'+k]));assert exact==d['cases'][0]['comparisons'][k]['exact'];assert exact==(k!='center')
                    for m in ['old','new']:assert sha(v[m+'_'+k].tobytes())==d['cases'][0]['comparisons'][k][m+'_sha256']
        else:
            assert d['status']=='complete' and d['operator_calls']==100 and all(c['all_exact'] for c in d['cases'])
            assert [(c['batch'],c['length']) for c in d['cases']]==[(b,n) for b in op['batches'] for n in op['lengths']]
        summary.append({'phase':phase,'archive_sha256':receipt['sha256'],'files_verified':len(files),'operator_calls':d['operator_calls'],'cells':cells,'status':d['status'],'library_sha256':build['library']['sha256'],'vendor_sources_unchanged':True})
    return {'status':'verified','candidates':summary,'total_operator_calls':310,'model_calls':0,'full_attribution_calls':0,'scope':'Synthetic finite-operator controls. The third/fourth candidates warrant full-model pilots; no complete-DT speed/memory claim from these timings.'}

if __name__=='__main__':
    d=verify();(S/'fa_owner_evidence_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
