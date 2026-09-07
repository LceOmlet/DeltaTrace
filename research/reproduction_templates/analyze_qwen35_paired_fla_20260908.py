"""Verify real captures and finite mixed algebra; CPU diagnostic, no model calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,sys,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from finite_fla_chunk_reference import coefficients_from_native_capture,exp_secant

def read_evidence(directory):
    d=A/'snapshot/tmp'/directory
    with zipfile.ZipFile(d/'review_bundle.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert Path(name).name==name;(d/name).write_bytes(z.read(name))
    r=json.loads((d/'results.json').read_bytes());p=json.loads((d/'protocol.json').read_bytes());assert p==r['protocol']
    for name,digest in p['files_sha256'].items():assert sha((d/name).read_bytes())==digest
    for item in r.get('artifacts',[]):assert sha((d/item['file']).read_bytes())==item['sha256']
    return d,r

def metric(reference,value):
    x=np.asarray(reference,dtype=np.float64);y=np.asarray(value,dtype=np.float64)
    assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    diff=y-x;den=float(np.linalg.norm(x.ravel()))
    return {'relative_L2':float(np.linalg.norm(diff.ravel())/den) if den else None,
            'max_abs':float(np.max(np.abs(diff))),'reference_norm':den}

SD,sr=read_evidence('codex_qwen35_official_spans_20260908_v1')
D,r=read_evidence('codex_qwen35_paired_fla_capture_20260908_v1')
assert sr['status']=='official_spans_remapped' and len(sr['cases'])==2
assert r['status']=='real_paired_endpoints_and_native_adjoints_captured',r.get('error')
assert r['protocol']['spans_sha256']==sha((SD/'results.json').read_bytes())
assert r['model_loads']==r['model_load_attempts']==r['root_forward_attempts']==r['root_forwards_completed']==1
assert r['adjoint_stage_attempts']==r['adjoint_stages_completed']==2
assert r['generation_calls']==r['quality_queries']==r['whole_model_attributions']==0
assert r['native_dispatch_counts']['FLA']==r['native_dispatch_counts']['conv']==24
assert r['native_dispatch_counts'].get('FA_dense',0)+r['native_dispatch_counts'].get('FA_varlen',0)==8
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['batch_shape']==[4,605] and r['valid_lengths']==[605,605,368,368]
kernel_counts=r['native_adjoint_profile']['kernel_counts']
assert any('chunk_bwd_kernel_dv' in k for k in kernel_counts)
assert any('chunk_gated_delta_rule_bwd_kernel_dhu' in k for k in kernel_counts)
def arrays(name):
    with np.load(D/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].astype(np.float64) for k in z.files}
endpoints=arrays('real_paired_FLA_prefix');adjoints=arrays('native_input_adjoints')
tick=time.perf_counter();coeff,lu=coefficients_from_native_capture(endpoints,adjoints,r['native_scale'])
tile_seconds=time.perf_counter()-tick

# Independent evaluation of the five local contractions in the stated finite
# recurrence. Use native baseline boundary/write values and native lambda_u,
# preserving their numerical errors. Temporary states are bounded to one
# 64-token chunk of one head on CPU, never used for model forward or scoring.
tick=time.perf_counter()
direct={k:np.empty_like(v) for k,v in coeff.items()}
for b in range(2):
    e0,e1=2*b,2*b+1
    for h in range(32):
        for block,start in enumerate(range(0,129,64)):
            stop=min(start+64,129)
            state=endpoints['h'][e0,block,h].copy();states=[];decayed=[]
            for t in range(start,stop):
                states.append(state.copy())
                C=np.exp(endpoints['raw_g'][e0,t,h])*state;decayed.append(C)
                state=C+np.outer(endpoints['k'][e0,t,h],endpoints['v_new'][e0,t,h])
                direct['q'][b,t,h]=r['native_scale']*(state@adjoints['do'][b,t,h])
            lam=adjoints['dh_end'][b,block,h].copy()
            for t in range(stop-1,start-1,-1):
                do=adjoints['do'][b,t,h];k=endpoints['k'][e1,t,h];u=lu[b,t,h]
                beta=endpoints['beta'][e1,t,h];C=decayed[t-start]
                lam=lam+np.outer(endpoints['q'][e1,t,h]*r['native_scale'],do)
                lr=-beta*u;lamC=lam+np.outer(k,lr)
                direct['v'][b,t,h]=beta*u
                direct['k'][b,t,h]=lam@endpoints['v_new'][e0,t,h]+C@lr
                direct['beta'][b,t,h]=(endpoints['v'][e0,t,h]-endpoints['k'][e0,t,h]@C)@u
                direct['alpha'][b,t,h]=np.sum(states[t-start]*lamC)
                direct['g'][b,t,h]=direct['alpha'][b,t,h]*exp_secant(endpoints['raw_g'][e0,t,h],endpoints['raw_g'][e1,t,h])
                lam=np.exp(endpoints['raw_g'][e1,t,h])*lamC
direct_seconds=time.perf_counter()-tick
comparisons={key:metric(direct[key],coeff[key]) for key in coeff}
effects=[]
for b,c in enumerate(sr['cases']):
    e0,e1=2*b,2*b+1
    value=float(np.sum(adjoints['do'][b]*(endpoints['o'][e1]-endpoints['o'][e0])))
    parts={key:float(np.sum(coeff[key][b]*(endpoints['raw_g' if key=='g' else key][e1]-endpoints['raw_g' if key=='g' else key][e0])))
           for key in ['q','k','v','beta','g']}
    estimate=sum(parts.values());absolute_terms=sum(abs(x) for x in parts.values())
    head_effect=np.sum(adjoints['do'][b]*(endpoints['o'][e1]-endpoints['o'][e0]),axis=(0,2))
    head_sum=np.zeros(32)
    for key in ['q','k','v','beta','g']:
        delta=endpoints['raw_g' if key=='g' else key][e1]-endpoints['raw_g' if key=='g' else key][e0]
        term=coeff[key][b]*delta
        head_sum+=np.sum(term,axis=(0,2) if term.ndim==3 else 0)
    assert np.isclose(head_effect.sum(),value) and np.isclose(head_sum.sum(),estimate)
    effects.append({'dataset':c['dataset'],'index':c['index'],'native_output_finite_effect':value,
        'coefficient_delta_sum':estimate,'parts':parts,'residual':estimate-value,
        'relative_abs_effect_residual':abs(estimate-value)/abs(value) if value else None,
        'residual_over_abs_branch_sum':abs(estimate-value)/absolute_terms if absolute_terms else None,
        'per_head_effects':head_effect.tolist(),'per_head_coefficient_sums':head_sum.tolist(),
        'per_head_residual_metrics':metric(head_effect,head_sum)})
s={'status':'CPU_mixed_finite_coefficients_on_real_native_endpoints_evaluated',
   'raw_sha256':{'spans':sha((SD/'results.json').read_bytes()),'paired_capture':sha((D/'results.json').read_bytes())},
   'coefficient_reference_sha256':sha((R/'research/runtime/finite_fla_chunk_reference.py').read_bytes()),
   'analyzer_sha256':sha(Path(__file__).read_bytes()),'mapping_cases':sr['cases'],
   'budget':{'CPU_text_only_records':2,'model_loads':1,'model_forwards':1,'physical_endpoint_batch':4,
             'native_adjoint_stage_calls':2,'complete_backward_calls':0,'generation_calls':0,'quality_queries':0,'whole_model_attributions':0},
   'native_dispatch_counts':r['native_dispatch_counts'],'native_adjoint_kernel_events':sum(kernel_counts.values()),
   'native_adjoint_profile_scope':'Profile hash and counts recorded remotely; full trace remains remote. Compact captured arrays and source/protocol identities independently verified locally.',
   'tile_vs_direct_mixed_contractions':comparisons,'local_finite_effects':effects,
   'CPU_diagnostic_seconds':{'tile':tile_seconds,'direct_chunk_oracle':direct_seconds},
   'native_diagnostic_costs':{k:r[k] for k in ['loading_seconds','forward_diagnostic_seconds','forward_diagnostic_peak_bytes',
       'adjoint_diagnostic_seconds','adjoint_diagnostic_peak_bytes','job_seconds']},
   'limits':['Only layer0, first129 tokens, two official examples; not whole model or answer-target attribution.',
      'Coefficients are CPU64 attribution references; no production GPU mixed kernel or FA-speed claim.',
      'Native BF16 boundary/WY/output rounding is retained; no post-hoc numerical pass threshold.',
      'Q/K coefficients act on normalized operands. L2, convolution, gates and remaining model pullbacks are pending.',
      'Prefix covers two chunk boundaries but no right-padded positions; model B4 padding executed, finite padded pullback unverified.']}
(A/'qwen35_paired_finite_summary_20260908.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'effects':effects,'costs':s['native_diagnostic_costs'],'CPU_seconds':s['CPU_diagnostic_seconds']}))
