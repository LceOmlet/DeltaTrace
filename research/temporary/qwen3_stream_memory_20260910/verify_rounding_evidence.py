"""Verify both failed rollout evidence and the exact but insufficient repair."""
from pathlib import Path
import hashlib,io,json,statistics,zipfile
import numpy as np
S=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def verify():
    receipt=json.loads((S/'rounding_archive_verification.json').read_bytes())
    archive=S/receipt['archive'];assert sha(archive.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('archive_manifest.json'));files={}
        assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
        for name,r in manifest['files'].items():
            data=z.read(name);assert sha(data)==r['sha256'] and len(data)==r['bytes'];files[name]=data
    def read(name):return json.loads(files[name])
    keys=['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']
    cells=[]
    for kind,lengths in [('rollout',[10,100,500]),('short',[128,256,512,1024])]:
        d=read('rounding_repair/'+kind+'/results.json');p=read('rounding_'+kind+'_protocol.json')
        assert d['status']=='complete' and d['protocol_sha256']==sha(files['rounding_'+kind+'_protocol.json'])
        assert d['driver_sha256']==p['driver_sha256']==sha(files['benchmark_rounding_'+kind+'.py'])
        assert len(d['cases'])==len(lengths) and len(d['rows'])==len(lengths)*5
        for n,h in p['repair_modules'].items():assert sha(files[n])==h
        with np.load(io.BytesIO(files['rounding_repair/'+kind+'/vectors.npz'])) as vectors:
            for case,n in zip(d['cases'],lengths):
                key='output_length' if kind=='rollout' else 'input_length';assert case[key]==n
                vec=vectors[str(n)];old=vectors[str(n)+'_retained'];assert np.array_equal(vec,old)
                assert sha(vec.tobytes())==case['audit']['vector_sha256'] and np.isfinite(vec).all()
                assert all(case['details'][k]==case['retained_details'][k] for k in keys)
                assert case['details']['deferred_validation']['all_passed']
                assert case['details']['deferred_validation']['predicates'] in [829,865]
                rows=[r for r in d['rows'] if r['target_'+key.replace('length','tokens')]==n]
                assert len(rows)==5 and all(r['status']=='ok' for r in rows)
                measured=[r for r in rows if r['phase']=='measured'];assert len(measured)==3
                cells.append({'kind':kind,'length':n,'vector_and_math_exact':True,
                    'mean_seconds':statistics.mean(r['time_sec'] for r in measured),
                    'cold_and_warm_peak_allocated_gb':max(r['peak_allocated_gb'] for r in rows),
                    'cold_and_warm_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in rows)})
    boundaries=read('rollout_boundaries/results.json')['cases'][0]['boundary_comparisons']
    assert len(boundaries)==616
    failed=[r for r in boundaries if not r['exact']]
    assert len(failed)==36 and {r['name'] for r in failed}=={'norm_two'}
    diagnostic=read('rollout_diagnostic/results.json')
    assert diagnostic['status']=='complete' and len(diagnostic['cases'])==3
    for case in diagnostic['cases']:
        assert not case['retained_vector_exact'] and case['graph_vs_direct_new']['vector_exact']
        assert all(case['graph_vs_direct_new']['math_fields'].values())
    queue=read('rollout/queue.json');assert len(queue['jobs'])==9
    incomplete_marker=[];bad=[];good=[]
    for job in queue['jobs']:
        path='rollout/'+job['name']+'/results.json';d=read(path)
        if job['status']!=d['status']:incomplete_marker.append({'name':job['name'],'queue_status':job['status'],'result_status':d['status']})
        if job['method']=='deltatrace_streamed':
            assert d['status']=='failed' and not d['cases'][0]['retained_vector_exact'];bad.append(job['name'])
        else:
            assert d['status']=='complete' and len(d['rows'])==5 and all(r['status']=='ok' for r in d['rows']);good.append(job['name'])
    assert len(bad)==3 and len(good)==6 and len(incomplete_marker)==1
    return {'status':'verified','archive_sha256':receipt['sha256'],'files_verified':len(files),'repaired_cells':cells,
        'repair_API_calls':49,'repaired_vectors_and_math_exact':True,'failed_original_curve_points':bad,
        'complete_original_FT_points':good,'paused_queue_marker_not_rewritten':incomplete_marker,
        'same_operand_boundary_comparisons':len(boundaries),'failed_norm_two_boundaries':len(failed),
        'repair_is_final_accepted_runtime':False,
        'remaining_cost_failure':'Repaired short1024 reserved19.675480064GB exceeds prior FT maximum19.662897152GB; rollout500 allocated18.696048128 and reserved19.539165184GB exceed FT17.860983808/18.087936GB. Costs are retained; this repair alone is insufficient.'}

if __name__=='__main__':
    result=verify();(S/'rounding_evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n');print(json.dumps(result,indent=2))
