"""Close the single contraction-reuse screen using saved outputs; no GPU calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from finite_fla_chunk_reference import coefficients_from_native_capture
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1'
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        path=(D/name).resolve();assert path.is_relative_to(D.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw);p=json.loads((D/'protocol.json').read_bytes())
assert r['protocol']==p
for name,digest in p['files_sha256'].items():assert sha((D/name).read_bytes())==digest
for item in r['artifacts']+r['compiler_generated_sources']:assert sha((D/item['file']).read_bytes())==item['sha256']
assert r['status']=='three_GEMM_reuse_screen_complete',r.get('error')
assert r['finite_pullback_attempts']==r['compiled_pullback_attempts']==5 and r['native_backward_attempts']==4
assert all(r[k]==0 for k in ['model_loads','model_forwards','generation_calls','quality_queries','whole_model_attributions','cached_adjoint_mixed_attempts'])
assert r['sources_before']==r['sources_after']
assert r['compiler_counters']['stats']['unique_graphs']==2 and not r['compiler_counters'].get('graph_break')
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1'
prior=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_compiled_20260908_v1'
assert sha((parent/'results.json').read_bytes())==p['parent_sha256']
assert sha((prior/'results.json').read_bytes())==p['prior_GPU_sha256']
assert (prior/'finite_fla_gpu.py').read_bytes()==(D/'finite_fla_gpu_baseline.py').read_bytes()
for name,digest in p['input_npz_sha256'].items():assert sha((parent/name).read_bytes())==digest
tree=ast.parse((A/'analyze_qwen35_finite_gpu_20260908.py').read_bytes())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load','metrics','effects']]
assert len(functions)==3
exec(compile(ast.Module(body=functions,type_ignores=[]),'<unchanged-output-analysis>','exec'))
ep=load(parent/'real_paired_FLA_prefix.npz');adj=load(parent/'native_input_adjoints.npz')
scale=json.loads((parent/'results.json').read_bytes())['native_scale']
reference,_=coefficients_from_native_capture(ep,adj,scale)
outputs={kind:load(D/(kind+'_coefficients.npz')) for kind in ['compiled','finite','native']}
formal=[row for row in r['calls'] if row['phase'].startswith('formal_')]
assert len(formal)==9 and all(x['status']=='complete' and not x['profiled'] for x in formal)
labels=p['route_labels']
timing={labels[kind]:{key:float(np.median([x[key] for x in formal if x['kind']==kind]))
    for key in ['wall_ms','event_ms','before_bytes','peak_bytes','after_bytes']} for kind in labels}
pairs=[]
for phase in ['formal_0','formal_1','formal_2']:
    rows={x['kind']:x for x in formal if x['phase']==phase};assert set(rows)==set(labels)
    pairs.append({'phase':phase,'wall_ms':{labels[k]:v['wall_ms'] for k,v in rows.items()},
        'candidate_over_baseline':rows['compiled']['wall_ms']/rows['finite']['wall_ms'],
        'candidate_over_native':rows['compiled']['wall_ms']/rows['native']['wall_ms']})
profiled={x['kind']:x for x in r['calls'] if x['profiled']};assert set(profiled)=={'compiled','finite'}
profiles={}
for kind,row in profiled.items():
    assert row['native_dispatch']=={'native_dv_local':1,'native_state_adjoint':1}
    prof=row['profile'];kc=prof['kernel_counts'];dur=prof['kernel_total_us']
    assert any('_finite_decay_scan' in n for n in kc)
    profiles[labels[kind]]={'wall_ms':row['wall_ms'],'kernel_events':sum(kc.values()),'kernel_us':sum(dur.values()),
        'top_kernels':sorted([{'name':n,'us':t,'count':kc[n]} for n,t in dur.items()],key=lambda x:-x['us'])[:10],
        'profile_sha256':prof['sha256']}
generated='\n'.join((D/x['file']).read_text() for x in r['compiler_generated_sources'])
assert 'extern_kernels.bmm_dtype' in generated
graph_bmm_counts={x['file']:(D/x['file']).read_text().count('extern_kernels.bmm_dtype(')
    for x in r['compiler_generated_sources'] if 'extern_kernels.bmm_dtype(' in (D/x['file']).read_text()}
assert sorted(graph_bmm_counts.values())==[15,18]
s={'status':'single_three_GEMM_reuse_candidate_analyzed_whole_model_pending','raw_sha256':sha(raw),
    'protocol_sha256':sha((D/'protocol.json').read_bytes()),'runtime_sha256':p['files_sha256']['finite_fla_gpu.py'],
    'archived_baseline_byte_identical':True,'native_sources_unchanged':True,'route_labels':labels,
    'candidate_vs_baseline':{k:metrics(outputs['finite'][k],outputs['compiled'][k]) for k in outputs['compiled']},
    'candidate_vs_CPU64':{k:metrics(reference[k],outputs['compiled'][k]) for k in outputs['compiled']},
    'candidate_finite_effects':effects(outputs['compiled']),'baseline_finite_effects':effects(outputs['finite']),
    'same_job_timing_medians':timing,'paired_ratios':pairs,
    'median_candidate_over_baseline':float(np.median([x['candidate_over_baseline'] for x in pairs])),
    'median_candidate_over_native':float(np.median([x['candidate_over_native'] for x in pairs])),
    'profiles':profiles,'compiler_counters':r['compiler_counters'],'compiler_generated_sources':r['compiler_generated_sources'],
    'generated_graph_BF16_bmm_call_counts':graph_bmm_counts,
    'cold_call_seconds':{labels[x['kind']]:x['wall_ms']/1000 for x in r['calls'] if x['phase']=='cold'},
    'job_seconds':r['job_seconds'],'budget':{'candidate_finite_calls':5,'archived_compiled_baseline_calls':5,
        'native_normalized_backward_helper_calls':4,'model_calls':0,'quality_queries':0,'generation_calls':0,'whole_model_attributions':0},
    'limitations':[p['cost_limit'],p['precision'],p['stop'],
        'The first formal candidate call and separate diagnostic profile are slower than the baseline and remain in the record.',
        'Three paired observations at B2,T129,H32,K128 do not establish stable general speedup or whole-model quality.']}
(A/'qwen35_finite_reuse_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({k:s[k] for k in ['candidate_vs_baseline','candidate_vs_CPU64','same_job_timing_medians','paired_ratios','compiler_counters','cold_call_seconds']},ensure_ascii=False))
print(json.dumps({'head_residuals':{key:[x['per_head_residual'] for x in s[key]] for key in ['candidate_finite_effects','baseline_finite_effects']},
    'profiles':{k:{n:v[n] for n in ['wall_ms','kernel_events','kernel_us']} for k,v in profiles.items()}}))
