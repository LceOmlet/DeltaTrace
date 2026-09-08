"""Saved native Q/K: separate finite normalization scale, radial feedback and precision.

CPU algebra only, no substitute native forward and no candidate/scorer. All
contractions are at the same token and compact key-head coordinates.
"""
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
def close(a,b,label):
    error=float((a-b).abs().max());assert error<1e-7,(label,error)
    return error
try:
    def stop(*a):raise TimeoutError('Frozen CPU-only norm audit budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['budget']['wall_time_seconds'])
    for name,h in p['files_sha256'].items():assert sha(A/name)==h
    src=p['source'];assert sha(src['result'])==src['result_sha256'] and sha(src['private'])==src['private_sha256']
    for item in p['native_source_receipts']:assert sha(item['path'])==item['sha256']
    source=json.loads(Path(src['result']).read_bytes());prior=json.loads(Path(p['scalar_result']).read_bytes())
    assert sha(p['scalar_result'])==p['scalar_result_sha256']
    assert source['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
    import torch,numpy as np
    from signed_secant_rules import rmsnorm_secant_pullback
    torch.set_num_threads(4);private=torch.load(src['private'],map_location='cpu',mmap=True,weights_only=True)
    st=private['1'];m=st['coeff'];n=st['native'];coeff=m['coeff'];eps=1e-6;vectors={}
    assert not torch.cuda.is_initialized()
    for key in ['q','k']:
        saved=m['m'+key].double();B,T,H,D=saved.shape;repeat=n['B2']['c']['raw_'+key].shape[2]//H
        def compact(x):
            x=x.double().reshape(B,T,H,repeat,D)
            assert torch.equal(x,x[:,:,:,:1].expand_as(x))
            return x[:,:,:,0]
        f=coeff[key].double().reshape(B,T,H,repeat,D).sum(3)
        x0,x1=[compact(n['B2']['c']['raw_'+key][i:i+1]) for i in [0,1]]
        radius=lambda x:(x.square().sum(-1)+eps).sqrt()
        r0,r1=radius(x0),radius(x1);bar=(x0+x1)/2
        scale=(1/r0+1/r1)/2;radial=-2/(r0*r1*(r0+r1))
        finite=scale[...,None]*f+radial[...,None]*bar*(bar*f).sum(-1,keepdim=True)
        proof=rmsnorm_secant_pullback(x0,x1,D**-.5,f,eps/D)
        close(finite,proof,'direct L2 vs existing RMS finite rule')
        r.setdefault('endpoint',{})[key]={'raw_shape':list(x0.shape),'repeat':repeat,
            'saved_coefficient_vs_CPU64_relative_L2':float((finite-saved).norm()/saved.norm()),
            'endpoint_real_closure_max':close((finite*(x1-x0)).sum(-1),(f*(x1/r1[...,None]-x0/r0[...,None])).sum(-1),'endpoint L2 real closure')}
        for step in p['steps']:
            if step=='B2':
                left={g:{k:v[1:2] for k,v in row.items()} for g,row in n['B2'].items()}
                right={g:{k:v[0:1] for k,v in row.items()} for g,row in n['B2'].items()}
            else:left,right=n['0'],n[step]
            xc,xa=[compact(x['c']['raw_'+key]) for x in [left,right]]
            nc,na=[compact(x['e'][key]) for x in [left,right]]
            rc,ra=radius(xc),radius(xa);delta=xc-xa;conditional_bar=(xc+xa)/2
            conditional_scale=(1/rc+1/ra)/2;conditional_radial=-2/(rc*ra*(rc+ra))
            fd=(f*delta).sum(-1)
            fixed_scale=scale*fd
            fixed_radial=radial*(f*bar).sum(-1)*(bar*delta).sum(-1)
            actual_scale=conditional_scale*fd
            actual_radial=conditional_radial*(f*conditional_bar).sum(-1)*(conditional_bar*delta).sum(-1)
            native=(coeff[key].double()*(left['e'][key].double()-right['e'][key].double())).sum(-1).reshape(B,T,H,repeat).sum(-1)
            close(native,(f*(nc-na)).sum(-1),'native head folding')
            analytic=(f*(xc/rc[...,None]-xa/ra[...,None])).sum(-1)
            close(analytic,actual_scale+actual_radial,'actual conditional finite algebra')
            prediction=(saved*delta).sum(-1)
            fields={'prediction':prediction,'actual_native':native,'analytic_native_input_normalization':analytic,
                'fixed_scale_response':fixed_scale,'fixed_radial_response':fixed_radial,
                'actual_scale_response':actual_scale,'actual_radial_response':actual_radial,
                'coefficient_precision':prediction-fixed_scale-fixed_radial,
                'inverse_scale_mismatch':fixed_scale-actual_scale,'radial_feedback_mismatch':fixed_radial-actual_radial,
                'native_normalization_difference':analytic-native,'error':prediction-native,
                'r0':r0,'r1':r1,'rC':rc,'rA':ra,
                'endpoint_cosine':(x0*x1).sum(-1)/(x0.norm(dim=-1)*x1.norm(dim=-1)).clamp_min(1e-30),
                'actual_cosine':(xc*xa).sum(-1)/(xc.norm(dim=-1)*xa.norm(dim=-1)).clamp_min(1e-30)}
            direction=x1-x0;length2=direction.square().sum(-1)
            t=((xa-x0)*direction).sum(-1)/length2.clamp_min(1e-30)
            off=xa-x0-t[...,None]*direction
            fields.update(partial_chord_position=t,partial_off_chord_norm=off.norm(dim=-1),
                relative_radial_change=(rc-ra)/rc,
                actual_delta_radial_fraction=((delta*xc).sum(-1).square()/(delta.square().sum(-1)*xc.square().sum(-1)).clamp_min(1e-30)))
            terms=['coefficient_precision','inverse_scale_mismatch','radial_feedback_mismatch','native_normalization_difference']
            closure=close(sum(fields[k] for k in terms),fields['error'],'four term closure')
            original=prior['points'][step]['QK'][key]['error']['net']
            assert abs(float(fields['error'].sum())-original)<1e-7
            row={'terms':{k:stats(fields[k]) for k in terms},'responses':{k:stats(fields[k]) for k in fields if k.endswith('response')},
                'error':stats(fields['error']),'closure_max':closure,'groups':{}}
            for group,sl in [('prompt',slice(0,source['input']['prompt_length'])),('response',slice(source['input']['prompt_length'],T))]:
                row['groups'][group]={k:stats(fields[k][:,sl]) for k in ['error',*terms]}
            order=fields['error'].abs().reshape(-1).argsort(descending=True);row['top_token_head_rows']=[]
            for index in order[:12].tolist():
                ti,hi=divmod(index,H)
                row['top_token_head_rows'].append({'token_position':ti,'compact_head':hi,**{k:float(v[0,ti,hi]) for k,v in fields.items()}})
            row['coordinates']='All response and residual fields share [sample, token, compact key head]; this is a native norm boundary, not original source-token attribution.'
            r['points'].setdefault(step,{})[key]=row
            for name,value in fields.items():vectors[step+'_'+key+'_'+name]=value.numpy()
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha(A/'vectors.npz')
    assert sha(src['private'])==src['private_sha256'] and not torch.cuda.is_initialized()
    for item in p['native_source_receipts']:assert sha(item['path'])==item['sha256']
    r['status']='MH2_GDN1_QK_native_norm_scale_radial_CPU_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).exists():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
