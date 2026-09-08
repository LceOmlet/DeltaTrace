"""CPU contraction/geometry on actual saved GDN1 states. No model/operator forward replacement."""
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
    def stop(*a):raise TimeoutError('Frozen CPU-only budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['budget']['wall_time_seconds'])
    for n,h in p['files_sha256'].items():assert sha(A/n)==h
    src=p['source'];assert sha(src['result'])==src['result_sha256'] and sha(src['private'])==src['private_sha256']
    source=json.loads(Path(src['result']).read_bytes());assert source['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
    import torch,numpy as np
    torch.set_num_threads(4)
    private=torch.load(src['private'],map_location='cpu',weights_only=True,mmap=True);st=private['1'];m=st['coeff'];n=st['native'];f=m['coeff']
    assert all(x.device.type=='cpu' for x in f.values()) and not torch.cuda.is_initialized()
    a0,a1=n['B2']['c']['a'][0:1].double(),n['B2']['c']['a'][1:2].double()
    g0,g1=n['B2']['e']['raw_g'][0:1].double(),n['B2']['e']['raw_g'][1:2].double()
    alpha0,alpha1=g0.exp(),g1.exp();da=a1-a0;dg=g1-g0
    vectors={};r['endpoint']={'a_equal':int((da==0).sum()),'g_equal':int((dg==0).sum()),'coordinates':da.numel()}
    # Alpha is a mathematical per-step exp(raw_g), not a newly captured native chunk intermediate.
    # Split existing core at this identity to detect adjacent compensation; never claim this is a native replay.
    r['alpha_scope']='CPU64 exp of actual native raw_g; analytic per-step decay coordinate for algebra only. Native FLA chunk uses cumulative gates; no native output is regenerated here.'
    for step in ['3','10','20','B2']:
        left={g:{k:v[1:2] for k,v in values.items()} for g,values in n['B2'].items()} if step=='B2' else n['0']
        right={g:{k:v[0:1] for k,v in values.items()} for g,values in n['B2'].items()} if step=='B2' else n[step]
        delta_a=left['c']['a'].double()-right['c']['a'].double();delta_g=left['e']['raw_g'].double()-right['e']['raw_g'].double()
        delta_alpha=left['e']['raw_g'].double().exp()-right['e']['raw_g'].double().exp()
        map_error=m['ma'].double()*delta_a-f['g'].double()*delta_g
        exp_error=f['g'].double()*delta_g-f['alpha'].double()*delta_alpha
        composite=map_error+exp_error
        row={'raw_g_map':stat(map_error),'exp_map_analytic':stat(exp_error),'a_to_alpha_composite':stat(composite)}
        original=source['layers']['1']['ledgers'][step]['internal']['terms'];assert abs(float(map_error.sum())-original['GDN_raw_g_parameter_map'])<1e-7
        row['recurrence_without_analytic_exp']=original['GDN_FLA_including_raw_g_exp']-float(exp_error.sum())
        row['FLA_plus_parameter_map']=original['GDN_FLA_including_raw_g_exp']+float(map_error.sum())
        assert abs(row['recurrence_without_analytic_exp']+float(composite.sum())-row['FLA_plus_parameter_map'])<1e-7
        outside=(right['c']['a'].double()<torch.minimum(a0,a1))|(right['c']['a'].double()>torch.maximum(a0,a1))
        row['raw_a_outside_endpoint_interval']={'coordinates':int(outside.sum()),'map_error':stat(map_error[outside]),'composite_error':stat(composite[outside])}
        ratio=torch.where(da!=0,(right['c']['a'].double()-a0)/torch.where(da!=0,da,torch.ones_like(da)),torch.zeros_like(da))
        row['endpoint_parameterization']={'coincident_a_with_different_g':int(((da==0)&(dg!=0)).sum()),
            'finite_slope_unique_on_noncoincident_scalar_endpoints':True,'partial_a_fraction_min':float(ratio[da!=0].min()),'partial_a_fraction_max':float(ratio[da!=0].max())}
        row['QK']={}
        for key in ['q','k']:
            raw=left['c']['raw_'+key].double()-right['c']['raw_'+key].double();norm=left['e'][key].double()-right['e'][key].double()
            mraw=m['m'+key].double();B,T,H,D=mraw.shape;repeat=raw.shape[2]//H
            raw=raw.reshape(B,T,H,repeat,D);assert (raw==raw[:,:,:,:1]).all()
            predicted=(mraw*raw[:,:,:,0]).sum(-1);actual=(f[key].double()*norm).sum(-1).reshape(B,T,H,repeat).sum(-1)
            error=predicted-actual
            x0=n['B2']['c']['raw_'+key][0:1].double().reshape(B,T,H,repeat,D)[:,:,:,0]
            x1=n['B2']['c']['raw_'+key][1:2].double().reshape(B,T,H,repeat,D)[:,:,:,0]
            xa=right['c']['raw_'+key].double().reshape(B,T,H,repeat,D)[:,:,:,0]
            line=x1-x0;distance=line.square().sum(-1);position=((xa-x0)*line).sum(-1)/distance.clamp_min(1e-30)
            off=xa-x0-position[...,None]*line;off_norm=off.square().sum(-1).sqrt()
            row['QK'][key]={'error':stat(error),'off_line_distance_mean':float(off_norm.mean()),'off_line_distance_max':float(off_norm.max()),
                'absolute_error_weighted_off_line_distance':float((error.abs()*off_norm).sum()/error.abs().sum().clamp_min(1e-30)),
                'fraction_coordinate_outside_segment':float(((position<0)|(position>1)).double().mean()),'scope':'Geometry of partial native raw Q/K relative to B2 endpoint line. This does not identify a replacement Jacobian.'}
            vectors[step+'_'+key+'_error']=error.numpy();vectors[step+'_'+key+'_off_line_distance']=off_norm.numpy()
        assert abs(sum(row['QK'][key]['error']['net'] for key in ['q','k'])-original['GDN_QK_L2_and_head_fold'])<1e-7
        for name,value in [('raw_g_map',map_error),('exp_map',exp_error),('composite',composite),('a_outside',outside)]:vectors[step+'_'+name]=value.numpy()
        r['points'][step]=row
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha(A/'vectors.npz')
    assert sha(src['private'])==src['private_sha256'] and sha(src['result'])==src['result_sha256'] and not torch.cuda.is_initialized()
    r['status']='MH2_GDN1_actual_scalar_composition_CPU_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-start
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).exists():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'points':r['points'],'error':r.get('error')}),flush=True)
