"""Offline independent finite identity checks; no model or GPU calls.

The FP64 row-block reference below is a mathematical diagnostic only. It is not
used for attribution, timing, a substitute model forward/backward, or quality.
Heads0/15 of both saved cases are fixed before reading candidate outcomes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_saved_20260908_v1'
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name;(D/name).write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw)
assert r['status']=='BF16_D256_right_padded_finite_FA_executed_on_original_decoder_operands'
for name,digest in r['protocol']['files_sha256'].items():assert sha((D/name).read_bytes())==digest
for artifact in r['artifacts']:
    if artifact['download']:
        f=D/artifact['file'];assert f.stat().st_size==artifact['bytes'] and sha(f.read_bytes())==artifact['sha256']
def metric(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));difference=b-a
    return {'relative_L2':float(np.linalg.norm(difference))/na if na else None,
            'max_abs':float(np.abs(difference).max()),'cosine':float(a@b)/(na*nb) if na*nb else None,'reference_norm':na}
ops=dict(np.load(D/'actual_operands.npz',allow_pickle=False))
actual=dict(np.load(D/'finite_actual.npz',allow_pickle=False))
equal=dict(np.load(D/'finite_equal.npz',allow_pickle=False))
native=dict(np.load(D/'native_gradients.npz',allow_pickle=False))
reduced={k:v if k=='dq' else v.reshape(2,4,4,605,256).sum(2,dtype=np.float32) for k,v in equal.items() if k in native}
checks={k:metric(native[k],reduced[k]) for k in native}
for k in checks:
    assert abs(checks[k]['relative_L2']-r['equal_endpoint_vs_native'][k]['relative_L2'])<1e-10
delta={k:ops[k[1]+'1'].astype(np.float64)-ops[k[1]+'0'].astype(np.float64) for k in ('dq','dk','dv')}
for k in ('dk','dv'):delta[k]=np.repeat(delta[k],4,axis=1)
terms={k:(actual[k].astype(np.float64)*delta[k]).sum((-1,-2)) for k in delta}
allocated=sum(terms.values())
direct=((ops['output1'].astype(np.float64)-ops['output0'].astype(np.float64))*ops['u'].astype(np.float64)).sum((-1,-2))
assert np.max(np.abs(direct-np.asarray(r['actual_finite_effect']['native_output_per_head'])))<1e-7
assert np.max(np.abs(allocated-np.asarray(r['actual_finite_effect']['allocated_per_head'])))<1e-7
assert all(np.count_nonzero(v[1,:,368:])==0 for group in (actual,equal,native) for v in group.values())
tick=time.perf_counter();oracle=[]
for batch,n in enumerate((605,368)):
    for head in (0,15):
        kh=head//4
        q0=ops['q0'][batch,head,:n].astype(np.float64);q1=ops['q1'][batch,head,:n].astype(np.float64)
        k0=ops['k0'][batch,kh,:n].astype(np.float64);k1=ops['k1'][batch,kh,:n].astype(np.float64)
        v0=ops['v0'][batch,kh,:n].astype(np.float64);u=ops['u'][batch,head,:n].astype(np.float64)
        lse0=ops['lse0'][batch,head,:n].astype(np.float64);lse1=ops['lse1'][batch,head,:n].astype(np.float64)
        ref={k:np.zeros((n,256),dtype=np.float64) for k in ('dq','dk','dv')}
        ref.update(tau=np.zeros(n),center=np.zeros(n))
        for a in range(0,n,32):
            z=min(a+32,n);s0=q0[a:z]@k0.T*0.0625;s1=q1[a:z]@k1.T*0.0625
            lp0=s0-lse0[a:z,None];lp1=s1-lse1[a:z,None];d=np.abs(lp1-lp0)
            ratio=np.divide(-np.expm1(-d),d,out=np.ones_like(d),where=d!=0)
            causal=np.arange(n)[None,:]<=np.arange(a,z)[:,None]
            lm=np.exp(np.maximum(lp0,lp1))*ratio*causal;P1=np.exp(lp1)*causal
            content=u[a:z]@v0.T;tau=lm.sum(1);center=(lm*content).sum(1)/tau
            ds=lm*(content-center[:,None])
            ref['dq'][a:z]=ds@((k0+k1)*0.5)*0.0625
            ref['dk']+=ds.T@((q0[a:z]+q1[a:z])*0.5)*0.0625
            ref['dv']+=P1.T@u[a:z]
            ref['tau'][a:z]=tau;ref['center'][a:z]=center
        oracle.append({'batch':batch,'head':head,'valid_length':n,
            'coefficients':{k:metric(ref[k],actual[k][batch,head,:n]) for k in ref}})
summary={'status':'BF16_D256_finite_FA_core_validated_decoder_propagation_pending',
    'raw_sha256':sha(raw),'protocol_sha256':sha((D/'protocol.json').read_bytes()),
    'source_sha256':r['protocol']['files_sha256'],'library_sha256':r['protocol']['library_sha256'],
    'native_decoder_replay':r['decoder_replay_vs_root'],'native_public_auxiliary':r['public_FA_output_vs_default_model_FA'],
    'native_calls':r['capture_calls'],'public_LSE_layout':r['public_LSE_layout'],
    'equal_endpoint_vs_native':checks,'finite_effect':{'direct_per_head':direct.tolist(),'allocated_per_head':allocated.tolist(),
        'terms':{k:v.tolist() for k,v in terms.items()},'head_vectors':[metric(direct[i],allocated[i]) for i in range(2)],
        'direct_sample_totals':direct.sum(1).tolist(),'allocated_sample_totals':allocated.sum(1).tolist(),
        'total_vector':metric(direct.sum(1),allocated.sum(1))},
    'CPU_reference':{'purpose':'Independent FP64 finite-rule diagnostic, native public LSE,32-query blocks; not a model or runtime backend.',
        'fixed_heads':[0,15],'rows_per_block':32,'seconds':time.perf_counter()-tick,'results':oracle},
    'padding_max_abs':r['padding_max_abs'],'calls':r['calls'],'artifacts':r['artifacts'],
    'source_tree_before':r['sources_before'],'source_tree_after':r['sources_after'],
    'versions':r['versions'],'actual_FA_arguments':r['actual_public_FA_arguments'],
    'family_budget':{'compiler_attempts':2,'successful_builds':1,'meta_model_constructions':2,'decoder_loads':2,
        'decoder_forward_calls':2,'auxiliary_public_FA_forwards':2,'public_FA_backwards':1,'finite_calls':3,
        'finite_kernel_launches':9,'whole_model_forwards':0,'whole_model_attributions':0,'generation_calls':0,'quality_queries':0},
    'limits':['One first full-attention layer, two previously used official cases, local output-derived cotangent.',
        'No complete attention input pullback: Q/K RMS, partial RoPE, sigmoid gate and projections are still pending.',
        'No full decoder/MLP/final answer target propagation, quality or complete attribution cost claim.',
        'Pinned finite framework FA2.5.3 differs from installed default model FA2.6.3; no universal ABI compatibility claim.',
        'One warm finite-core timing does not establish stable speed or a fair comparison to native backward/FT.']}
(A/'qwen35_finite_fa_actual_summary_20260908.json').write_text(json.dumps(summary,indent=2),encoding='utf-8',newline='\n')
print(json.dumps({k:summary[k] for k in ['status','native_decoder_replay','native_public_auxiliary','equal_endpoint_vs_native','padding_max_abs']}))
print(json.dumps({'finite_head_errors':[v['relative_L2'] for v in summary['finite_effect']['head_vectors']],
                  'total_finite_error':summary['finite_effect']['total_vector']['relative_L2'],
                  'CPU_reference_seconds':summary['CPU_reference']['seconds']}))
