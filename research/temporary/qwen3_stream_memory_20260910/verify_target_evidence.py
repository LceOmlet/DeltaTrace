"""Independently recompute all accepted repair vectors and retain failed seeds."""
from pathlib import Path
import hashlib,io,json,statistics,zipfile
import numpy as np
S=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
KEYS=['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']

def archive(phase):
    receipt=json.loads((S/(phase+'_archive_verification.json')).read_bytes());p=S/receipt['archive']
    assert sha(p.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(p) as z:
        manifest=json.loads(z.read('archive_manifest.json'));assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
        files={n:z.read(n) for n in manifest['files']}
        assert all(sha(files[n])==r['sha256'] and len(files[n])==r['bytes'] for n,r in manifest['files'].items())
    return files,receipt

def cases(files,folder,prefix,kind,lengths):
    p=json.loads(files[prefix+'_protocol.json']);d=json.loads(files[folder+'/results.json'])
    assert d['status']=='complete' and d['driver_sha256']==p['driver_sha256']
    assert d['protocol_sha256']==sha(files[prefix+'_protocol.json'])
    for n,h in p['repair_modules'].items():assert sha(files[n])==h
    assert len(d['cases'])==len(lengths) and len(d['rows'])==len(lengths)*5
    result=[]
    with np.load(io.BytesIO(files[folder+'/vectors.npz'])) as v:
        for c,n in zip(d['cases'],lengths):
            assert c[kind+'_length']==n
            vec=v[str(n)];old=v[str(n)+'_retained'];assert np.array_equal(vec,old) and np.isfinite(vec).all()
            assert sha(vec.tobytes())==c['audit']['vector_sha256']
            assert all(c['details'][k]==c['retained_details'][k] for k in KEYS)
            assert c['details']['deferred_validation']['all_passed']
            rows=[r for r in d['rows'] if r['target_'+kind+'_tokens']==n];assert len(rows)==5 and all(r['status']=='ok' for r in rows)
            measured=[r for r in rows if r['phase']=='measured'];assert len(measured)==3
            result.append({'folder':folder,'length':n,'full_vector_and_math_exact':True,'mean_seconds':statistics.mean(r['time_sec'] for r in measured),'all_call_peak_allocated_gb':max(r['peak_allocated_gb'] for r in rows),'all_call_peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in rows)})
    return result

def verify():
    target,tr=archive('target');headroom,hr=archive('headroom');read=lambda n:json.loads(target[n])
    verified=cases(target,'exact_target/rollout','exact_target_rollout','output',[10,100,500])+cases(target,'exact_target/short','exact_target_short','input',[128,256,512,1024])+cases(headroom,'headroom','headroom','input',[1024])
    failed=read('bounded_target/rollout/results.json');assert failed['status']=='failed'
    assert failed['cases'][0]['retained_vector_exact'] and not failed['cases'][1]['retained_vector_exact']
    with np.load(io.BytesIO(target['bounded_target/rollout/vectors.npz'])) as v:
        assert np.array_equal(v['10'],v['10_retained']) and not np.array_equal(v['100_new'],v['100_retained'])
    bd=read('bounded_diagnostic/results.json');assert bd['status']=='complete';c=bd['cases'][0]
    boundaries=c['boundary_comparisons'];bad=[r for r in boundaries if not r['exact']]
    assert len(boundaries)==618 and len(bad)==1 and bad[0]['name']=='bounded_seed.0'
    return {'status':'verified','archives':{'target':tr['sha256'],'headroom':hr['sha256']},'exact_cells':verified,
        'accepted_candidate_API_calls':56,'full_program_bounded_seed_rejected':True,'same_operand_boundaries':len(boundaries),'failed_boundaries':bad,
        'final_runtime_claim':False,'scope':'Numerical evidence for exact-target/headroom candidates. Subsequent public-runtime v2 speed gate is separate and failed.'}

if __name__=='__main__':
    d=verify();(S/'target_evidence_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
