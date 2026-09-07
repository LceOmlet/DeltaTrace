"""Audit actual official-compiler outputs, generated sources and matched costs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from finite_fla_chunk_reference import coefficients_from_native_capture
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_compiled_20260908_v1'
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        path=(D/name).resolve();assert path.is_relative_to(D.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw);p=json.loads((D/'protocol.json').read_bytes());assert p==r['protocol']
for name,digest in p['files_sha256'].items():assert sha((D/name).read_bytes())==digest
for item in r['artifacts']:assert sha((D/item['file']).read_bytes())==item['sha256']
for item in r['compiler_generated_sources']:assert sha((D/item['file']).read_bytes())==item['sha256']
assert r['status']=='official_compiler_fused_mixed_finite_screen_complete',r.get('error')
assert r['finite_pullback_attempts']==r['compiled_pullback_attempts']==5 and r['native_backward_attempts']==4
assert all(r[k]==0 for k in ['model_loads','model_forwards','generation_calls','quality_queries','whole_model_attributions','cached_adjoint_mixed_attempts'])
assert r['sources_before']==r['sources_after']
assert r['compiler_counters']['stats']['unique_graphs']==1
assert not r['compiler_counters'].get('graph_break')
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1'
prior=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_20260908_v1'
assert sha((parent/'results.json').read_bytes())==p['parent_sha256']
assert sha((prior/'results.json').read_bytes())==p['prior_GPU_sha256']
for name,digest in p['input_npz_sha256'].items():assert sha((parent/name).read_bytes())==digest
tree=ast.parse((A/'analyze_qwen35_finite_gpu_20260908.py').read_bytes())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load','metrics','effects']]
assert len(functions)==3
exec(compile(ast.Module(body=functions,type_ignores=[]),'<same-output-analysis>','exec'))
ep=load(parent/'real_paired_FLA_prefix.npz');adj=load(parent/'native_input_adjoints.npz')
scale=json.loads((parent/'results.json').read_bytes())['native_scale']
reference,_=coefficients_from_native_capture(ep,adj,scale)
outputs={kind:load(D/(kind+'_coefficients.npz')) for kind in ['compiled','finite','native']}
def code_functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
old,new=code_functions(prior/'finite_fla_gpu.py'),code_functions(D/'finite_fla_gpu.py')
assert set(new)-set(old)=={'make_compiled_finite_pullback'} and all(old[n]==new[n] for n in old)
formal=[row for row in r['calls'] if row['phase'].startswith('formal_')];assert len(formal)==9
assert all(row['status']=='complete' and not row['profiled'] for row in formal)
timing={kind:{key:float(np.median([row[key] for row in formal if row['kind']==kind]))
    for key in ['wall_ms','event_ms','before_bytes','peak_bytes','after_bytes']} for kind in ['compiled','finite','native']}
pairs=[]
for phase in ['formal_0','formal_1','formal_2']:
    rows={x['kind']:x for x in formal if x['phase']==phase}
    pairs.append({'phase':phase,'compiled_over_eager':rows['compiled']['wall_ms']/rows['finite']['wall_ms'],
                  'compiled_over_native':rows['compiled']['wall_ms']/rows['native']['wall_ms']})
profiled={x['kind']:x for x in r['calls'] if x['profiled']};assert set(profiled)=={'compiled','finite'}
profiles={}
for kind,row in profiled.items():
    assert row['native_dispatch']=={'native_dv_local':1,'native_state_adjoint':1}
    prof=row['profile'];kc=prof['kernel_counts'];dur=prof['kernel_total_us']
    profiles[kind]={'kernel_events':sum(kc.values()),'kernel_us':sum(dur.values()),
       'top_kernels':sorted([{'name':n,'us':t,'count':kc[n]} for n,t in dur.items()],key=lambda x:-x['us'])[:10],
       'profile_sha256':prof['sha256']}
    assert any('_finite_decay_scan' in n for n in kc)
generated='\n'.join((D/x['file']).read_text() for x in r['compiler_generated_sources'])
assert 'extern_kernels.bmm_dtype' in generated
assert any('triton_' in n for n in profiled['compiled']['profile']['kernel_counts'])
s={'status':'official_compiler_fusion_verified_on_real_finite_FLA_inputs','raw_sha256':sha(raw),
   'protocol_sha256':sha((D/'protocol.json').read_bytes()),'GPU_source_sha256':p['files_sha256']['finite_fla_gpu.py'],
   'source_math_unchanged_from_first_GPU':True,
   'compiled_vs_eager':{k:metrics(outputs['finite'][k],outputs['compiled'][k]) for k in outputs['compiled']},
   'compiled_vs_CPU64':{k:metrics(reference[k],outputs['compiled'][k]) for k in outputs['compiled']},
   'finite_effects':effects(outputs['compiled']),
   'same_job_timing_medians':timing,'paired_ratios':pairs,
   'median_compiled_over_eager':float(np.median([x['compiled_over_eager'] for x in pairs])),
   'median_compiled_over_native':float(np.median([x['compiled_over_native'] for x in pairs])),
   'profiles':profiles,'compiler_counters':r['compiler_counters'],
   'compiler_generated_sources':r['compiler_generated_sources'],
   'compile_cold_call_seconds':r['calls'][0]['wall_ms']/1000,'job_seconds':r['job_seconds'],
   'budget':{'compiled_finite_calls':5,'eager_finite_calls':5,'native_normalized_backward_helper_calls':4,
             'model_calls':0,'quality_queries':0,'whole_model_attributions':0},
   'limitations':[p['cost_limit'],'Static fullgraph tested only at actual B2,T129,H32,K128, including final tile padding. New lengths/shapes may recompile.',
     'max_autotune=False does not disable all compiler tuning: the actual counter records51 GPU benchmarking calls in compilation.',
     '45 kernel launches still exceed the native helper; no whole-model FA-speed or quality claim. Whole-model nonlinear and full-attention integration pending.']}
(A/'qwen35_finite_gpu_compiled_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({'parity':s['compiled_vs_eager'],'head_residuals':[x['per_head_residual'] for x in s['finite_effects']],
                 'timing':timing,'ratios':pairs,'profiles':profiles},ensure_ascii=False))
