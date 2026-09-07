"""Read captured GPU outputs and matched costs; no model or GPU calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from finite_fla_chunk_reference import coefficients_from_native_capture
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_20260908_v1'
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name;(D/name).write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw);p=json.loads((D/'protocol.json').read_bytes());assert r['protocol']==p
for name,digest in p['files_sha256'].items():assert sha((D/name).read_bytes())==digest
for item in r['artifacts']:assert sha((D/item['file']).read_bytes())==item['sha256']
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1'
assert sha((parent/'results.json').read_bytes())==p['parent_sha256']
for name,digest in p['input_npz_sha256'].items():assert sha((parent/name).read_bytes())==digest
assert r['status']=='GPU_mixed_finite_and_matched_native_backward_executed',r.get('error')
assert all(r[k]==0 for k in ['model_loads','model_forwards','generation_calls','quality_queries','whole_model_attributions'])
assert r['finite_pullback_attempts']==r['native_backward_attempts']==5 and r['cached_adjoint_mixed_attempts']==1
assert r['sources_before']==r['sources_after']
assert all(row['status']=='complete' for row in r['calls'])
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].astype(np.float64) for k in z.files}
ep=load(parent/'real_paired_FLA_prefix.npz');adj=load(parent/'native_input_adjoints.npz')
scale=json.loads((parent/'results.json').read_bytes())['native_scale']
reference,_=coefficients_from_native_capture(ep,adj,scale)
outputs={kind:load(D/(kind+'_coefficients.npz')) for kind in ['cached','finite','native']}
def metrics(x,y):
    assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    den=float(np.linalg.norm(x.ravel()));err=float(np.linalg.norm((y-x).ravel()))
    return {'relative_L2':err/den if den else None,'max_abs':float(np.max(np.abs(x-y))),'reference_norm':den}
def effects(coeff):
    result=[]
    for b in range(2):
        e0,e1=2*b,2*b+1
        actual=(adj['do'][b]*(ep['o'][e1]-ep['o'][e0])).sum(axis=(0,2))
        total=np.zeros(32);parts={}
        for key in ['q','k','v','beta','g']:
            name='raw_g' if key=='g' else key
            term=coeff[key][b]*(ep[name][e1]-ep[name][e0]);head=term.sum(axis=(0,2) if term.ndim==3 else 0)
            total+=head;parts[key]=float(head.sum())
        result.append({'dataset':['niah_mq_q2','morehopqa'][b],'index':[0,1][b],
            'per_head_actual_effect':actual.tolist(),'per_head_coefficient_effect':total.tolist(),
            'per_head_residual':metrics(actual,total),'branch_sums':parts,
            'total_effect':float(actual.sum()),'total_coefficient_effect':float(total.sum())})
    return result
numerics={kind:{key:metrics(reference[key],value) for key,value in outputs[kind].items()} for kind in ['cached','finite']}
profiled={row['kind']:row for row in r['calls'] if row['profiled']}
assert set(profiled)=={'finite','native'}
assert profiled['finite']['native_dispatch']=={'native_dv_local':1,'native_state_adjoint':1}
assert profiled['native']['native_dispatch']=={'native_backward':1,'native_state_reconstruction':1,'native_dv_local':1,
                                             'native_state_adjoint':1,'native_WY_backward':1}
assert any('_finite_decay_scan' in name for name in profiled['finite']['profile']['kernel_counts'])
formal=[x for x in r['calls'] if x['phase'].startswith('formal_')];assert len(formal)==6
timing={kind:{metric:float(np.median([row[metric] for row in formal if row['kind']==kind]))
              for metric in ['wall_ms','event_ms','before_bytes','peak_bytes','after_bytes']} for kind in ['finite','native']}
ratios=[next(row['wall_ms'] for row in formal if row['kind']=='finite' and row['phase']==phase)/
        next(row['wall_ms'] for row in formal if row['kind']=='native' and row['phase']==phase) for phase in ['formal_0','formal_1','formal_2']]
profiles={}
for kind,row in profiled.items():
    pc=row['profile'];names=pc['kernel_counts'];dur=pc['kernel_total_us']
    profiles[kind]={'kernel_events':sum(names.values()),'kernel_us':sum(dur.values()),
                   'top_kernels_by_time':sorted([{'name':name,'us':time,'count':names[name]} for name,time in dur.items()],key=lambda x:-x['us'])[:12],
                   'source_profile_sha256':pc['sha256'],'scope':'Full profile retained remotely; compact kernel counts/times recorded by driver.'}
s={'status':'GPU_mixed_finite_local_screen_complete','raw_sha256':sha(raw),'protocol_sha256':sha((D/'protocol.json').read_bytes()),
   'GPU_source_sha256':p['files_sha256']['finite_fla_gpu.py'],'parent_sha256':p['parent_sha256'],
   'numerics_vs_CPU64_same_saved_adjoints':numerics['cached'],'numerics_vs_CPU64_with_fresh_native_adjoints':numerics['finite'],
   'cached_vs_fresh_adjoints_GPU_outputs':{k:metrics(outputs['cached'][k],outputs['finite'][k]) for k in outputs['cached']},
   'finite_V_vs_native_backward_V':metrics(outputs['native']['v'],outputs['finite']['v']),
   'finite_effects':effects(outputs['finite']),'CPU64_effects':effects(reference),
   'matched_local_timing_medians':timing,'paired_wall_ratios':ratios,'median_paired_wall_ratio':float(np.median(ratios)),
   'profiles':profiles,'resident_input_bytes':r['resident_input_bytes'],'job_seconds':r['job_seconds'],
   'budget':{'cached_mixed_calls':1,'full_local_finite_calls':5,'native_normalized_backward_helper_calls':5,
             'model_calls':0,'quality_queries':0,'generation_calls':0,'whole_model_attributions':0},
   'limitations':[p['cost_limit'],'Only layer0 real prefixes of129 tokens, two historical official examples, one fixed local cotangent.',
     'CPU64 is a finite algebra reference; this is not a new precision requirement or a bitwise correctness gate.',
     'Three paired timings are a local screen, not a steady whole-model speed claim or benchmark quality result.']}
(A/'qwen35_finite_gpu_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({'errors':numerics['finite'],'head_errors':[x['per_head_residual'] for x in s['finite_effects']],
                  'timing':timing,'ratios':ratios,'profile':profiles},ensure_ascii=False))
