"""One frozen analytic L2-path candidate on saved native K states, CPU only."""
import os,json,time,hashlib,traceback,signal,zipfile
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());started=time.perf_counter()
r={'status':'starting','protocol':p,'model_calls':0,'GPU_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'points':{}}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum()),'max_absolute':float(x.abs().max())}
try:
    def stop(*a):raise TimeoutError('Frozen CPU-only K path budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['budget']['wall_time_seconds'])
    for name,h in p['files_sha256'].items():assert sha(A/name)==h
    src=p['source'];assert sha(src['result'])==src['result_sha256'] and sha(src['private'])==src['private_sha256']
    assert sha(p['geometry_result'])==p['geometry_result_sha256']
    geometry=json.loads(Path(p['geometry_result']).read_bytes());assert geometry['status']=='MH2_GDN1_QK_native_norm_scale_radial_CPU_complete'
    import torch,numpy as np
    from integrated_l2_path_20260909 import integrated_l2_pullback
    torch.set_num_threads(4);private=torch.load(src['private'],map_location='cpu',mmap=True,weights_only=True)
    st=private['1'];m=st['coeff'];n=st['native'];saved=m['mk'].double();B,T,H,D=saved.shape
    repeat=n['B2']['c']['raw_k'].shape[2]//H
    def compact(x):
        x=x.double().reshape(B,T,H,repeat,D)
        assert torch.equal(x,x[:,:,:,:1].expand_as(x))
        return x[:,:,:,0]
    x0,x1=[compact(n['B2']['c']['raw_k'][i:i+1]) for i in [0,1]]
    f=m['coeff']['k'].double().reshape(B,T,H,repeat,D).sum(3)
    candidate=integrated_l2_pullback(x0,x1,f)
    # Single precision at the real repeated-head coefficient boundary, then fold.
    expanded=integrated_l2_pullback(n['B2']['c']['raw_k'][0:1].float(),n['B2']['c']['raw_k'][1:2].float(),m['coeff']['k'].float())
    candidate32=expanded.reshape(B,T,H,repeat,D).sum(3).double()
    assert torch.isfinite(candidate32).all() and not torch.cuda.is_initialized()
    radius=lambda x:(x.square().sum(-1,keepdim=True)+1e-6).sqrt()
    endpoint_target=(f*(x1/radius(x1)-x0/radius(x0))).sum(-1)
    r['endpoint_real_closure_max']=float(((candidate*(x1-x0)).sum(-1)-endpoint_target).abs().max())
    assert r['endpoint_real_closure_max']<1e-7
    r['candidate32_vs64_relative_L2']=float((candidate32-candidate).norm()/candidate.norm())
    # Independent integral check on fixed evenly-spaced actual rows, never chosen by errors.
    indices=torch.linspace(0,T*H-1,p['quadrature_rows']).long()
    a,b,u=[x.reshape(-1,D)[indices] for x in [x0,x1,f]]
    nodes,weights=np.polynomial.legendre.leggauss(p['quadrature_nodes']);integral=torch.zeros_like(u)
    for node,weight in zip(nodes,weights):
        x=a+(float(node)+1)/2*(b-a);rr=(x.square().sum(-1,keepdim=True)+1e-6).sqrt()
        integral+=float(weight)/2*(u/rr-x*(x*u).sum(-1,keepdim=True)/rr.pow(3))
    exact=candidate.reshape(-1,D)[indices]
    r['independent_quadrature']={'rows':len(indices),'nodes':len(nodes),'max_absolute':float((integral-exact).abs().max()),'relative_L2':float((integral-exact).norm()/exact.norm()),'scope':'CPU mathematical Jacobian check on actual endpoint rows, not model/FA/FLA or a benchmark.'}
    assert r['independent_quadrature']['relative_L2']<1e-7
    vectors={}
    for step in p['steps']:
        if step=='B2':
            left={g:{k:v[1:2] for k,v in row.items()} for g,row in n['B2'].items()}
            right={g:{k:v[0:1] for k,v in row.items()} for g,row in n['B2'].items()}
        else:left,right=n['0'],n[step]
        delta=compact(left['c']['raw_k'])-compact(right['c']['raw_k'])
        native=(m['coeff']['k'].double()*(left['e']['k'].double()-right['e']['k'].double())).sum(-1).reshape(B,T,H,repeat).sum(-1)
        fields={'actual_native':native,'current_prediction':(saved*delta).sum(-1),'candidate_prediction':(candidate*delta).sum(-1),'candidate32_prediction':(candidate32*delta).sum(-1)}
        for name in ['current','candidate','candidate32']:fields[name+'_error']=fields[name+'_prediction']-native
        assert abs(float(fields['current_error'].sum())-geometry['points'][step]['k']['error']['net'])<1e-7
        r['points'][step]={'fields':{name:stats(x) for name,x in fields.items()},'groups':{g:{name:stats(x[:,sl]) for name,x in fields.items()} for g,sl in [('prompt',slice(0,480)),('response',slice(480,T))]}}
        for name,x in fields.items():vectors[step+'_'+name]=x.numpy()
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha(A/'vectors.npz')
    assert sha(src['private'])==src['private_sha256'] and not torch.cuda.is_initialized()
    r['status']='MH2_GDN1_K_analytic_path_CPU_scout_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).exists():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
