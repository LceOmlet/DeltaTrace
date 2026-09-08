"""NumPy audit of public scalar/normalization contractions; no model work."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_scalar_composition_20260909_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':x.sum(),'positive':x.clip(min=0).sum(),'negative':x.clip(max=0).sum(),'absolute':np.abs(x).sum()}
r,p=read(D/'results.json'),read(D/'protocol.json');assert r['protocol']==p and r['status']=='MH2_GDN1_actual_scalar_composition_CPU_complete'
assert sha(D/'protocol.json')=='5b537a0818c7b984d9dd2b0a173255de532e3609a9da0f3609bfa9225114e2dd'
assert p['files_sha256']['study.py']=='67fa0401bf705c673167b25c071bb4ac6825a624af7665a3cdfe7507fe6e0fd1'
receipt=read(D/'terminal_receipt.json');assert receipt['proc_exists'] is False
for n,x in receipt['files'].items():assert sha(D/n)==x['sha256'] and (D/n).stat().st_size==x['bytes']
for n,h in p['files_sha256'].items():assert sha(D/n)==h
with zipfile.ZipFile(D/'review_bundle.zip') as z:
 for n in z.namelist():assert not n.endswith('.pt') and z.read(n)==(D/n).read_bytes()
assert all(r[k]==0 for k in ['model_calls','GPU_calls','DT_calls','scorer_calls','FT_calls','generation_calls'])
S=A/'snapshot'/p['source']['result'].lstrip('/');source=read(S);assert sha(S)==p['source']['result_sha256']
v=np.load(D/'vectors.npz',allow_pickle=False);keys=set()
for step,row in r['points'].items():
 arrays={n:v[step+'_'+n] for n in ['raw_g_map','exp_map','composite','a_outside','q_error','k_error','q_off_line_distance','k_off_line_distance']};keys.update(step+'_'+n for n in arrays)
 for x in arrays.values():assert np.isfinite(x).all()
 close(arrays['raw_g_map']+arrays['exp_map'],arrays['composite'])
 for n,target in [('raw_g_map','raw_g_map'),('exp_map','exp_map_analytic'),('composite','a_to_alpha_composite')]:
  for k,x in stats(arrays[n]).items():close(x,row[target][k])
 mask=arrays['a_outside'];assert mask.dtype==bool and mask.sum()==row['raw_a_outside_endpoint_interval']['coordinates']
 for n,target in [('raw_g_map','map_error'),('composite','composite_error')]:
  for k,x in stats(arrays[n][mask]).items():close(x,row['raw_a_outside_endpoint_interval'][target][k])
 original=source['layers']['1']['ledgers'][step]['internal']['terms']
 close(arrays['raw_g_map'].sum(),original['GDN_raw_g_parameter_map'])
 close(row['recurrence_without_analytic_exp']+arrays['composite'].sum(),row['FLA_plus_parameter_map'])
 close(row['FLA_plus_parameter_map'],original['GDN_FLA_including_raw_g_exp']+original['GDN_raw_g_parameter_map'])
 close(arrays['q_error'].sum()+arrays['k_error'].sum(),original['GDN_QK_L2_and_head_fold'])
 for qk in ['q','k']:
  for k,x in stats(arrays[qk+'_error']).items():close(x,row['QK'][qk]['error'][k])
  dist=arrays[qk+'_off_line_distance'];close(dist.mean(),row['QK'][qk]['off_line_distance_mean']);close(dist.max(),row['QK'][qk]['off_line_distance_max'])
assert set(v.files)==keys
out={'status':'MH2_GDN1_scalar_composition_independent_CPU_audit_passed','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),
 'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'points':r['points'],'endpoint':r['endpoint'],'job_seconds':r['seconds'],
 'interpretation':'At mid10 raw-g map overprediction6.10264 is only partly offset by analytic exp term-.85091, leaving5.25173. Outside endpoint scalar coordinates contribute9.74988 raw-map net, while inside coordinates compensate. Key normalization is3.96545 of4.11149 QK error. Scalar finite slope is uniquely constrained; avoid arbitrary slope edits. Next study may regroup original input projection and scalar decay, requiring real full propagation and original metrics before acceptance.',
 'limits':['Analytic CPU64 alpha coordinate, not a native FLA replay or model counterfactual. No GPU/model/metric/candidate claim.','NumPy independently checks public contractions and identities; private activation reductions and tensor geometry were computed remotely. Signed magnitudes retained; totals can cancel.']}
path=A/'dt_MH2_GDN1_scalar_composition_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'sha256':sha(path)}))
