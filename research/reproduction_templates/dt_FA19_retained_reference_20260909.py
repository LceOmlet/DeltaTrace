"""Six public FA probes distinguish EOS, actual-clean and retained-content references.

All compared routing contrasts share output-query coordinates and fixed seed.
No fitted reference, replacement forward/backward, candidate or metric curve.
"""
import os,time,json,hashlib,signal,traceback,zipfile,gc
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca';A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());started=time.perf_counter()
r={'status':'starting','protocol':p,'native_FA_entered':0,'native_FA_returned':0,'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'backward_calls':0,'generation_calls':0,'cases':{},'calls':[]};torch=None
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(x):return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}
try:
    def timeout(*a):raise TimeoutError('Frozen six native-FA retained-reference probes expired')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_seconds'])
    for n,h in p['files_sha256'].items():assert sha(A/n)==h
    import torch,numpy as np,flash_attn.flash_attn_interface as fa
    torch.set_num_threads(4);assert sha(fa.__file__)==p['FA_interface_sha256'];vectors={}
    kwargs=dict(p['native_FA_kwargs']);kwargs['window_size']=tuple(kwargs['window_size'])
    for src in p['sources']:
        key=src['case'];assert sha(src['private_path'])==src['private_sha256'] and sha(src['reference_vectors'])==src['reference_vectors_sha256']
        private=torch.load(src['private_path'],map_location='cpu',weights_only=True,mmap=True)
        if key=='MH0':st=private;points={s:st['native'][s]['c'] for s in ['0','3','10']}
        elif key=='MH2':st=private['19'];points={s:st['native'][s]['c'] for s in ['0','3','10']}
        else:st=private['FA19'];points={s:st['B1'][s]['c'] for s in ['0','3','10']}
        seed=st['coeff']['mcontent'].transpose(1,2).double();ref=np.load(src['reference_vectors'],allow_pickle=False)
        def project(x):assert x.shape==seed.shape;return (seed*x.double()).sum((2,3)).squeeze(0)
        r['cases'][key]={'points':{},'source':src}
        for step in ['3','10']:
            ops=[points[step][n].transpose(1,2).to('cuda').contiguous() for n in ['query','key']]+[points['0']['value'].transpose(1,2).to('cuda').contiguous()]
            row={'case':key,'step':step,'status':'entered'};r['calls'].append(row);tick=time.perf_counter();r['native_FA_entered']+=1
            with torch.no_grad():hybrid=fa.flash_attn_func(*ops,**kwargs)
            torch.cuda.synchronize();r['native_FA_returned']+=1;row['status']='returned';row['seconds']=time.perf_counter()-tick
            oc,oa=points['0']['attention_output'],points[step]['attention_output'];hybrid=hybrid.to('cpu',copy=True)
            RC=project(oc.double()-hybrid.double());VA=project(hybrid.double()-oa.double());actual=project(oc.double()-oa.double())
            prefix='FA19_'+step+'_' if key=='MH3' else step+'_'
            R0=torch.from_numpy(ref[prefix+'R0']).double().reshape(-1);RA=torch.from_numpy(ref[prefix+'RA']).double().reshape(-1)
            fields={'R0':R0,'RA':RA,'RC':RC,'EOS_reference_error':R0-RA,'clean_reference_error':RC-RA,
                'actual_P_A_content':VA,'actual_total':actual,'joint_route_content_interaction':RC-RA}
            assert (RC+VA-actual).abs().max()<1e-7
            r['cases'][key]['points'][step]={'fields':{n:stats(v) for n,v in fields.items()},'closure_max':float((RC+VA-actual).abs().max())}
            for n,v in fields.items():vectors[key+'_'+step+'_'+n]=v.numpy()
            del hybrid,ops
        assert sha(src['private_path'])==src['private_sha256'] and sha(src['reference_vectors'])==src['reference_vectors_sha256']
        del private,st,points,seed,ref;gc.collect()
    assert r['native_FA_entered']==r['native_FA_returned']==6
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha(A/'vectors.npz')
    assert sha(fa.__file__)==p['FA_interface_sha256'];r['status']='three_FA19_retained_references_6publicFA_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    if torch is not None and torch.cuda.is_initialized():r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    r['scope']='R0,RA,RC are the same native routing change measured using EOS, actual retained, or actual clean payloads. Same seed/query coordinates make these contrast differences directly comparable. Hybrids are FA-operator probes, not full-model counterfactuals. No chosen reference is promoted and no source-token attribution claim follows.'
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).is_file():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
