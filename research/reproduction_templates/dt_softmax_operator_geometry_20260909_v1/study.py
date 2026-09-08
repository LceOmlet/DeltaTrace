"""CPU geometry audit of actual saved MH0 rows; no attention/model implementation.

Normalizes FP64 dot products of saved native BF16 operands for a mathematical
operator diagnostic, not a native FA replay or a benchmark quality score.
Only two preselected queries and all sixteen heads are inspected. No T*T array.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import hashlib,json,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch

A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());tick=time.perf_counter()
r={'status':'starting','protocol':p,'GPU_calls':0,'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'rows':[]}
arrays={}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def soft(x):
    e=np.exp(x-np.max(x));return e/e.sum()
def J(prob,g):return prob*(g-prob@g)
def stats(x):return {'net':float(x.sum()),'positive':float(x[x>0].sum()),'negative':float(x[x<0].sum()),'absolute':float(abs(x).sum())}
def vec(t):assert t.device.type=='cpu';return t.double().numpy()
try:
    for n,v in p['files_sha256'].items():assert digest(A/n)==v
    assert digest(p['private_artifact_path'])==p['private_artifact_sha256']
    assert digest(p['source_results_path'])==p['source_results_sha256']
    torch.set_num_threads(4)
    data=torch.load(p['private_artifact_path'],map_location='cpu',weights_only=True)
    paired=data['native']['B2']['c'];seed=data['coeff']['mcontent'].to(torch.bfloat16)
    assert paired['query'].shape==(2,16,853,256) and seed.shape==(1,16,853,256)
    for query in p['queries']:
      for head in range(16):
        prefix=f'q{query}_h{head}';kh=head//4;n=query+1
        def logits(c,ep):
            return vec(c['key'][ep,kh,:n])@vec(c['query'][ep,head,query])*p['scale']
        z0=logits(paired,0);z1=logits(paired,1);p0=soft(z0);p1=soft(z1);s=z1-z0;d=p1-p0
        g=vec(paired['value'][0,kh,:n])@vec(seed[0,head,query])
        D=float(d@s);v=J(p1,s);variance=float(s@v);base=J(p1,g);b=float(g@d)
        assert D>0 and variance>0 and np.all(p1>0)
        # Original ideal logmean matrix: symmetric PSD and secant.
        lp0=z0-np.logaddexp.reduce(z0);lp1=z1-np.logaddexp.reduce(z1);gap=lp1-lp0
        w=np.empty_like(p0);small=abs(gap)<1e-12
        w[small]=(p0[small]+p1[small])*.5
        w[~small]=(p1[~small]-p0[~small])/gap[~small]
        lm=w*(g-(w@g)/w.sum())
        supported=base+d*((b-float(base@s))/D)
        # Symmetric positive semidefinite secant matrix (BFGS-form update).
        symmetric=base-v*(float(base@s)/variance)+d*(b/D)
        # Inertia of supported's symmetric part via a 2x2 congruence.
        u=d/np.sqrt(p1);vv=np.sqrt(p1)*(s-p1@s);aa=float(u@u)
        e1=u/np.sqrt(aa);perp=vv-e1*(D/np.sqrt(aa));bn=float(np.linalg.norm(perp))
        e2=perp/bn if bn>0 else np.zeros_like(perp)
        block=np.array([[aa/D,-np.sqrt(aa)*bn/(2*D)],[-np.sqrt(aa)*bn/(2*D),1.]])
        eigen,eigenvectors=np.linalg.eigh(block);x=e1*eigenvectors[0,0]+e2*eigenvectors[1,0]
        witness=x/np.sqrt(p1);native_w=J(p1,witness)
        supported_w=native_w+d*((witness@d-native_w@s)/D)
        symmetric_w=native_w-v*((native_w@s)/variance)+d*((witness@d)/D)
        out={'query':query,'head':head,'valid_keys':n,'D':D,'endpoint_variance':variance,
            'supported_congruence_min_eigenvalue':float(eigen[0]),
            'witness_quadratic':{'native':float(witness@native_w),'supported':float(witness@supported_w),'symmetric':float(witness@symmetric_w)},
            'actual_g_quadratic':{k:float(g@m) for k,m in [('native',base),('logmean',lm),('supported',supported),('symmetric',symmetric)]},
            'secant_residual':{k:float(m@s-b) for k,m in [('logmean',lm),('supported',supported),('symmetric',symmetric)]},
            'zero_sum':{k:float(m.sum()) for k,m in [('logmean',lm),('supported',supported),('symmetric',symmetric)]},'conditions':{}}
        for label in ('3','10'):
            zc=logits(data['native']['0']['c'],0);za=logits(data['native'][label]['c'],0)
            actual=float(g@(soft(zc)-soft(za)));change=zc-za
            pred={k:float(m@change) for k,m in [('logmean',lm),('supported',supported),('symmetric',symmetric)]}
            out['conditions'][label]={'ideal_softmax_effect_at_saved_V0':actual,'predictions':pred,'errors':{k:y-actual for k,y in pred.items()}}
            arrays[prefix+'_z'+label]=za
        for name,value in {'z0':z0,'z1':z1,'zclean':zc,'g':g,'witness':witness}.items():arrays[prefix+'_'+name]=value
        r['rows'].append(out)
    r['query_summaries']={}
    for query in p['queries']:
        rows=[x for x in r['rows'] if x['query']==query]
        r['query_summaries'][str(query)]={'non_PSD_supported_rows':sum(x['supported_congruence_min_eigenvalue']<-1e-10 for x in rows),
            'negative_actual_g_quadratic':sum(x['actual_g_quadratic']['supported']<-1e-10 for x in rows),
            'conditions':{step:{method:stats(np.array([x['conditions'][step]['errors'][method] for x in rows]))
                              for method in ('logmean','supported','symmetric')} for step in ('3','10')}}
    np.savez_compressed(A/'rows.npz',**arrays);r['vectors_sha256']=digest(A/'rows.npz')
    assert digest(p['private_artifact_path'])==p['private_artifact_sha256']
    r['status']='actual_saved_rows_CPU_geometry_and_conditional_diagnostic_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['seconds']=time.perf_counter()-tick
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','rows.npz']:
            if (A/n).exists():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error'),'queries':r.get('query_summaries')}),flush=True)
