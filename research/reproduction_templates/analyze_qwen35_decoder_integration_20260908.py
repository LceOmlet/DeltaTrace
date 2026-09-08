"""Independently audit saved full-decoder vectors, without GPU/model execution."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        path=(D/name).resolve();assert path.is_relative_to(D.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw)
assert r['status']=='both_original_decoder_families_finite_propagation_executed'
for name,digest in r['protocol']['files_sha256'].items():
    data=(D/name).read_bytes();assert sha(data)==digest;ast.parse(data)
for item in r['artifacts']+r['compiler_generated_sources']:
    if item.get('download',True):
        f=D/item['file'];assert f.stat().st_size==item['bytes'] and sha(f.read_bytes())==item['sha256']
def metric(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,
        'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None,'reference_norm':na}
layers={}
for index in (3,0):
    row=r['layers'][str(index)];a=dict(np.load(D/f'layer{index}_actual.npz',allow_pickle=False))
    e=np.load(D/f'layer{index}_equal.npz',allow_pickle=False)['finite']
    n=np.load(D/f'layer{index}_native_gradient.npz',allow_pickle=False)['input_gradient']
    assert all(v.shape==(2,605,4096) and np.isfinite(v).all() for v in list(a.values())+[e,n])
    direct=(a['upstream'].astype(np.float64)*(a['output1'].astype(np.float64)-a['output0'].astype(np.float64))).sum((1,2))
    tokens=(a['finite'].astype(np.float64)*(a['input1'].astype(np.float64)-a['input0'].astype(np.float64))).sum(-1)
    allocated=tokens.sum(1);actual=row['effects']['decoder']
    assert np.max(np.abs(direct-actual['direct']))<1e-7 and np.max(np.abs(allocated-actual['allocated']))<1e-7
    limit=metric(n,e);assert abs(limit['relative_L2']-row['equal_endpoint_vs_native']['relative_L2'])<1e-10
    assert all(np.count_nonzero(v[1,368:])==0 for v in [a['finite'],a['upstream'],e,n])
    assert row['cold_warm_difference']['max_abs']==0
    assert all(v['max_abs']==0 for v in row['endpoint_boundaries'].values())
    layers[str(index)]={'block_type':row['block_type'],'native_replay_vs_root':row['native_replay_vs_root'],
        'equal_endpoint_vs_native':limit,'equal_endpoint_per_sample':[metric(n[b],e[b]) for b in range(2)],
        'finite_effect':{'direct':direct.tolist(),'allocated':allocated.tolist(),'residual':(allocated-direct).tolist(),
            'relative_residual_per_sample':((allocated-direct)/np.abs(direct)).tolist(),**metric(direct,allocated)},
        'local_input_position_contributions':{'positive_mass':np.maximum(tokens,0).sum(1).tolist(),
            'negative_mass':np.minimum(tokens,0).sum(1).tolist(),'limit':'Hidden-state allocations for a local output cotangent, not final input-token attribution.'},
        'padding_max_abs':0,'cold_warm_max_abs':0,'endpoint_boundaries':row['endpoint_boundaries'],
        'native_capture_calls':{'decoder':row['decoder_capture_calls'],'mixer':row['mixer_capture_calls']},
        'native_backward_node_events':row['native_backward_node_events'],'reported_boundary_effects':row['effects']}
assert r['sources_before']==r['sources_after']
assert r['finite_attempts']==6 and r['native_forward_attempts']==2 and r['native_backward_attempts']==2
assert r['full_model_forwards']==r['generation_calls']==r['quality_queries']==r['whole_model_attributions']==0
s={'status':'both_original_decoder_families_checked_final_target_and_whole_model_pending',
    'raw_sha256':sha(raw),'protocol_sha256':sha((D/'protocol.json').read_bytes()),'source_sha256':r['protocol']['files_sha256'],
    'layers':layers,'calls':r['calls'],'budget':r['protocol']['budget'],'job_seconds':r['job_seconds'],
    'source_tree':r['sources_before'],'versions':r['versions'],'compiler_counters':r['compiler_counters'],
    'compiler_benchmark_observations':r['compiler_benchmark_observations'],
    'benchmark_observation_limit':r['benchmark_observation_limit'],'compiler_generated_sources':r['compiler_generated_sources'],
    'artifacts':r['artifacts'],'limits':['Only original layers3 and0, official historical NI0/MH1 trajectories, local output cotangent.',
        'Full-vector equal-endpoint checks and finite effects do not establish whole-model quality or numerical stability.',
        'Cold original calls and warm finite calls are not a matched speed comparison; no FT or whole-model cost claim.',
        'Compiler-source hashes are provenance, not independent proof of mathematical correctness.',
        'Default FA/FLA precision retained; GDN decoder1.66% finite effect residual remains visible.']}
(A/'qwen35_decoder_integration_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8',newline='\n')
print(json.dumps({'status':s['status'],'layers':{k:{x:v[x] for x in ['equal_endpoint_vs_native','finite_effect','padding_max_abs']} for k,v in layers.items()},'compiler_sources':len(r['compiler_generated_sources'])}))
