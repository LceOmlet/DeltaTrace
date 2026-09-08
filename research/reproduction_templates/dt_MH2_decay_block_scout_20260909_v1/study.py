"""CPU64 local scout: endpoint-supported input Jacobian of x -> exp(g(Wa x))."""
import os,json,time,hashlib,traceback,signal,zipfile
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());start=time.perf_counter()
r={'status':'starting','protocol':p,'model_calls':0,'GPU_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'points':{}}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stat(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}
try:
    def stop(*a):raise TimeoutError('Frozen CPU scout budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(120)
    for n,h in p['files_sha256'].items():assert sha(A/n)==h
    src=p['source'];assert sha(src['result'])==src['result_sha256'] and sha(src['private'])==src['private_sha256']
    source=json.loads(Path(src['result']).read_bytes());assert source['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
    import torch,numpy as np
    from safetensors import safe_open
    torch.set_num_threads(4)
    st=torch.load(src['private'],map_location='cpu',weights_only=True,mmap=True)['1'];n=st['native'];m=st['coeff'];f=m['coeff']
    cp=Path(p['checkpoint']);index=json.loads((cp/'model.safetensors.index.json').read_bytes());weights={}
    r['tensor_sources']={}
    for name in ['in_proj_a.weight','A_log','dt_bias']:
        key='model.language_model.layers.1.linear_attn.'+name;file=cp/index['weight_map'][key]
        assert [file.stat().st_size,file.stat().st_mtime_ns]==p['expected_weight_stats'][file.name]
        with safe_open(file,framework='pt',device='cpu') as handle:weights[name]=handle.get_tensor(key).double()
        r['tensor_sources'][name]={'file':file.name,'key':key,'shape':list(weights[name].shape),'CPU64_tensor_sha256':hashlib.sha256(weights[name].numpy().tobytes()).hexdigest()}
    W,alog,bias=[weights[k] for k in ['in_proj_a.weight','A_log','dt_bias']]
    x0,x1=n['B2']['c']['input'][0:1].double(),n['B2']['c']['input'][1:2].double();dx=x1-x0
    g0,g1=n['B2']['e']['raw_g'][0:1].double(),n['B2']['e']['raw_g'][1:2].double();alpha0,alpha1=g0.exp(),g1.exp()
    a1=n['B2']['c']['a'][1:2].double();derivative=-alog.exp()*(a1+bias).sigmoid()*alpha1
    current=m['ma'].double()@W;jac=(f['alpha'].double()*derivative)@W
    target=(f['alpha'].double()*(alpha1-alpha0)).sum(-1,keepdim=True)
    residual=target-(jac*dx).sum(-1,keepdim=True);denom=dx.square().sum(-1,keepdim=True);assert (denom>0).all()
    candidate=jac+dx*(residual/denom)
    r['candidate_endpoint_closure_max']=float(((candidate*dx).sum(-1,keepdim=True)-target).abs().max());assert r['candidate_endpoint_closure_max']<1e-7
    r['coefficient_norms']={'current':float(current.norm()),'candidate':float(candidate.norm()),'Jac1':float(jac.norm()),'rank1_correction':float((candidate-jac).norm()),'dx_min_norm':float(denom.min().sqrt())}
    vectors={}
    for step in ['3','10','20','B2']:
        clean={g:{k:v[1:2] for k,v in vs.items()} for g,vs in n['B2'].items()} if step=='B2' else n['0']
        part={g:{k:v[0:1] for k,v in vs.items()} for g,vs in n['B2'].items()} if step=='B2' else n[step]
        change=clean['c']['input'].double()-part['c']['input'].double()
        effect=(f['alpha'].double()*(clean['e']['raw_g'].double().exp()-part['e']['raw_g'].double().exp())).sum(-1)
        pred0=(current*change).sum(-1);pred1=(candidate*change).sum(-1)
        projection_transfer=pred0-(m['ma'].double()*(clean['c']['a'].double()-part['c']['a'].double())).sum(-1)
        fields={'current_prediction':pred0,'candidate_prediction':pred1,'analytic_decay_effect':effect,'current_error':pred0-effect,'candidate_error':pred1-effect,'projection_transfer':projection_transfer,'candidate_minus_current':pred1-pred0}
        row={k:stat(v) for k,v in fields.items()};r['points'][step]=row
        for k,v in fields.items():vectors[step+'_'+k]=v.numpy()
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha(A/'vectors.npz')
    assert sha(src['private'])==src['private_sha256'] and not torch.cuda.is_initialized()
    r['status']='MH2_decay_block_CPU64_local_scout_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-start
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).exists():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'points':r['points'],'error':r.get('error')}),flush=True)
