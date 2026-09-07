"""Verify real-tensor finite FA operators without promoting failed capture."""
import ast,hashlib,json,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v2'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete'
assert p==json.loads((A/'vendor_fa_captured_operator_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_captured_operator_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==sha(A/name)==digest
source=A/'snapshot'/p['capture_results'].lstrip('/');assert sha(source)==p['capture_results_sha256']
captured=json.loads(source.read_text())
assert captured['status']=='failed' and not captured['operator_attempts']
assert not captured['records'][0]['signed_vector_exact_to_parent']
assert d['capture_exact_parent_guard']=='failed_in_predecessor_not_relaxed'
capture=captured['captured_layers'][0];assert capture==p['actual_operands']
assert d['native_sources_before']==d['native_sources_after']==p['native_source_sha256']
assert d['native_root_forwards']==d['native_vjps']==d['manual_passes']==0
build_path=A/'snapshot'/p['build_result'].lstrip('/');assert sha(build_path)==p['build_result_sha256']
build=json.loads(build_path.read_text());assert build['library']['sha256']==p['library_sha256']
failure=json.loads((A/'vendor_fa_capture_failure_summary_20260907.json').read_text())
assert failure['raw_sha256']==sha(source) and failure['parent_vector_guard']=='failed_preserved'

def load(meta):
    path=F/meta['file'];assert sha(path)==meta['sha256']
    x=np.load(path,allow_pickle=False);assert list(x.shape)==meta['shape'] and str(x.dtype)==meta['dtype'].removeprefix('torch.')
    return x
data={k:load(v) for k,v in capture['operands'].items()}
assert data['q0'].shape==(1,32,601,128) and capture['groups']==4
assert len(d['operator_attempts'])==14 and all(x['complete'] for x in d['operator_attempts'])
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'scope':p['purpose'],
     'capture_budget_spent_in_predecessor':failure['budget'],'failed_capture_preserved':failure,'operator_budget':p['operator_budget'],'costs':{},'arrays':{},'source_math_preserved_before_probe':True}
outputs={}
for kind,number in [('dense_P1',4),('finite_FA',5),('standard_FA',5)]:
    runs=[r for r in d['operator_attempts'] if r['kind']==kind];assert len(runs)==number
    measured=[r for r in runs if not r['warmup'] and not r['profile']];assert len(measured)==3
    first=runs[0];assert first['warmup']
    if kind!='standard_FA':outputs[kind]={k:load(v) for k,v in first['output_arrays'].items()}
    out['costs'][kind]={'median_seconds':statistics.median(r['seconds'] for r in measured),
        'max_incremental_peak_bytes':max(r['incremental_peak_bytes'] for r in measured),
        'max_total_peak_bytes':max(r['peak_bytes'] for r in measured),
        'all_repeats_identical':all(r['output_stats']==first['output_stats'] for r in runs),
        'warmup_seconds':first['seconds']}
    for r in runs:
        assert r['seconds']>0 and r['peak_bytes']==r['resident_bytes']+r['incremental_peak_bytes']
        if kind=='finite_FA':
            assert r['activity']['calls_attempted']==r['activity']['calls_enqueued']==1
            assert len(r['activity']['buffer_contract'])==15
            for buf in r['activity']['buffer_contract']:
                assert buf['shape'] in [[1,32,601,128],[1,32,601]],buf
out['profiles']={}
for kind,profile in d['profiles'].items():
    path=F/profile['trace'];assert sha(path)==profile['sha256']
    events=json.loads(path.read_text())['traceEvents'];kernels=[e['name'] for e in events if e.get('cat')=='kernel']
    assert sorted(e['name'] for e in events if e.get('cat') in ['kernel','gpu_memcpy'])==sorted(profile['GPU_kernels'])  # CUDA events include separately categorized host copies
    if kind=='finite_FA':assert sum('deltatrace_fa_finite_p1_kernel' in n for n in kernels)==3
    else:
        assert sum('flash_fwd_kernel' in n for n in kernels)==1
        assert any('flash_bwd' in n for n in kernels)
    out['profiles'][kind]={'GPU_kernel_count':len(kernels),'finite_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in kernels)}

def error(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    return {'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
            'maximum_absolute':float(np.max(np.abs(a-b))),'finite':bool(np.isfinite(a).all())}
for kind in ['dense_P1','finite_FA']:
    out['arrays'][kind]={key:error(outputs[kind][key],data['expected_'+key]) for key in ['dq','dk','dv']}

# CPU64 reference of the exact declared native-LSE finite formula, streamed by
# query blocks for this verification. It is a local mathematical oracle only.
q0,k0,q1,k1,v0,u=[data[k].astype(np.float64) for k in ['q0','k0','q1','k1','v0','u']]
lse0=data['lse0'].astype(np.float64);lse1=data['lse1'].astype(np.float64);scale=capture['scale']
qm=(q0+q1)*.5;km=(k0+k1)*.5;n=q0.shape[-2]
oracle={'dq':np.zeros_like(q0),'dk':np.zeros_like(q0),'dv':np.zeros_like(q0),
        'tau':np.zeros_like(lse0),'center':np.zeros_like(lse0)}
for start in range(0,n,64):
    end=min(n,start+64);urows=u[...,start:end,:]
    z0=(q0[...,start:end,:]@k0.swapaxes(-1,-2))*scale
    z1=(q1[...,start:end,:]@k1.swapaxes(-1,-2))*scale
    valid=np.arange(n)[None,:]<=np.arange(start,end)[:,None]
    lp0=z0-lse0[...,start:end,None];lp1=z1-lse1[...,start:end,None]
    distance=np.abs(lp1-lp0);ratio=np.ones_like(distance)
    np.divide(-np.expm1(-distance),distance,out=ratio,where=distance!=0)
    mean=np.where(valid,np.exp(np.maximum(lp0,lp1))*ratio,0.)
    t=urows@v0.swapaxes(-1,-2);tau=mean.sum(-1);center=(mean*t).sum(-1)/tau
    ds=mean*(t-center[...,None]);p1=np.where(valid,np.exp(lp1),0.)
    oracle['dq'][...,start:end,:]=ds@km*scale
    oracle['dk']+=ds.swapaxes(-1,-2)@qm[...,start:end,:]*scale
    oracle['dv']+=p1.swapaxes(-1,-2)@urows
    oracle['tau'][...,start:end]=tau;oracle['center'][...,start:end]=center
out['CPU64_native_LSE_rule']={kind:{key:error(outputs[kind][key],oracle[key]) for key in oracle} for kind in outputs}
for kind in outputs:
    values=outputs[kind]
    credit=sum(float(np.sum(values[key].astype(np.float64)*delta)) for key,delta in [('dq',q1-q0),('dk',k1-k0),('dv',data['v1'].astype(np.float64)-v0)])
    target=float(np.sum(u*(data['out1'].astype(np.float64)-data['out0'].astype(np.float64))))
    out.setdefault('local_signed_credit',{})[kind]={'credit':credit,'actual_native_output_delta_contraction':target,'unassigned':target-credit,
        'negative_elements':sum(int((values[key]<0).sum()) for key in ['dq','dk','dv'])}
out['local_numerical_guard_pass']=all(out['arrays']['finite_FA'][key]['relative_l2']<=p['predeclared_numerical_review']['max_relative_l2_to_frozen_dense_dq_dk_dv'] for key in ['dq','dk','dv'])
c=out['costs'];out['finite_to_dense_time_ratio']=c['finite_FA']['median_seconds']/c['dense_P1']['median_seconds']
out['finite_to_standard_FA_FB_time_ratio']=c['finite_FA']['median_seconds']/c['standard_FA']['median_seconds']
out['finite_to_dense_incremental_peak_ratio']=c['finite_FA']['max_incremental_peak_bytes']/c['dense_P1']['max_incremental_peak_bytes']

out['local_sign_rounding']={}
for key in ['dq','dk','dv']:
    x=outputs['finite_FA'][key].astype(np.float64);y=data['expected_'+key].astype(np.float64)
    zero=(y!=0)&(x==0);flip=x*y<0
    out['local_sign_rounding'][key]={'nonzero_to_zero':int(zero.sum()),
        'zeroed_reference_l2_fraction':float(np.linalg.norm(y[zero])/np.linalg.norm(y)),
        'opposite_nonzero_sign':int(flip.sum()),
        'flipped_reference_l2_fraction':float(np.linalg.norm(y[flip])/np.linalg.norm(y))}
v1=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v1/results.json'
failed=json.loads(v1.read_text());assert failed['status']=='failed'
assert len(failed['operator_attempts'])==1 and not failed['operator_attempts'][0]['complete']
assert failed['operator_attempts'][0]['error'].endswith("ModuleNotFoundError: No module named 'signed_secant_rules'\n")
out['earlier_packaging_failure']={'raw_sha256':sha(v1),'missing_dependency':'signed_secant_rules',
    'dense_attempted_not_executed':1,'finite_operator_calls':0,'model_calls':0,
    'elapsed_seconds':failed['elapsed_seconds'],'preserved':True}
out['limits']=['One actual original NI0 layer35, B1 H32 N601 D128; not end-to-end or long-sequence validation.',
    'Finite half intermediates and native LSE differ numerically from old explicit FP32 normalization.',
    'Profile CUDA kernels include wrapper and output-diagnostic kernels; only three are finite-extension kernels.',
    'Standard FA is native GQA forward+backward cost reference, not the finite mathematical reference.',
    'Ordinary FA repeated gradients are not bitwise identical; no repeat-stability claim for that reference.',
    'Dense local reference is minimal required P1 math with eager logarithmic mean, not a whole production core benchmark.']

(A/'vendor_fa_captured_operator_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
