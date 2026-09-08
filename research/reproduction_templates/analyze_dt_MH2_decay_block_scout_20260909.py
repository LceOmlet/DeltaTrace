"""No-model NumPy audit and rejection of the local endpoint-supported decay block."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_decay_block_scout_20260909_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':x.sum(),'positive':x.clip(min=0).sum(),'negative':x.clip(max=0).sum(),'absolute':np.abs(x).sum()}
r,p=read(D/'results.json'),read(D/'protocol.json');assert r['protocol']==p and r['status']=='MH2_decay_block_CPU64_local_scout_complete'
assert sha(D/'protocol.json')=='ccf59923c8e0ea7a3f4f364f3cca6d3e86d819dc290bcc6a067f6a3079dcda42'
assert p['files_sha256']['study.py']=='94e24f12668429e8850d8abdf1d2bdbe04170c7d5a22c31fc29d00595883e7b4'
receipt=read(D/'terminal_receipt.json');assert receipt['proc_exists'] is False
for n,x in receipt['files'].items():assert sha(D/n)==x['sha256'] and (D/n).stat().st_size==x['bytes']
for n,h in p['files_sha256'].items():assert sha(D/n)==h
with zipfile.ZipFile(D/'review_bundle.zip') as z:
 for n in z.namelist():assert not n.endswith('.pt') and z.read(n)==(D/n).read_bytes()
assert all(r[k]==0 for k in ['model_calls','GPU_calls','DT_calls','scorer_calls','FT_calls','generation_calls'])
S=A/'snapshot'/p['source']['result'].lstrip('/');assert sha(S)==p['source']['result_sha256']
C=read(A/'dt_MH2_GDN1_scalar_composition_summary_20260909.json')
v=np.load(D/'vectors.npz',allow_pickle=False);keys=set()
for step,row in r['points'].items():
 arrays={k:v[step+'_'+k] for k in row};keys.update(step+'_'+k for k in arrays)
 for n,x in arrays.items():
  assert x.shape==(1,710) and x.dtype==np.float64 and np.isfinite(x).all()
  for k,z in stats(x).items():close(z,row[n][k])
 close(arrays['current_prediction']-arrays['analytic_decay_effect'],arrays['current_error'])
 close(arrays['candidate_prediction']-arrays['analytic_decay_effect'],arrays['candidate_error'])
 close(arrays['candidate_prediction']-arrays['current_prediction'],arrays['candidate_minus_current'])
 close(arrays['current_error'].sum()-arrays['projection_transfer'].sum(),C['points'][step]['a_to_alpha_composite']['net'])
assert set(v.files)==keys
assert r['candidate_endpoint_closure_max']<1e-7
assert r['points']['10']['candidate_error']['net']>r['points']['10']['current_error']['net']>0
assert r['points']['10']['candidate_error']['absolute']>r['points']['10']['current_error']['absolute']
out={'status':'MH2_decay_block_CPU64_scout_independent_audit_passed_candidate_rejected','results_sha256':sha(D/'results.json'),
 'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),
 'job_seconds':r['seconds'],'points':r['points'],'coefficient_norms':r['coefficient_norms'],'tensor_sources':r['tensor_sources'],
 'decision':'Reject this original-input-Jacobian plus rank-one endpoint correction as a repair of the measured MH2 GDN1 mid mismatch; no production implementation or whole-metric expansion. Endpoint closure and small early aggregate improvement are insufficient; mid net error5.25004to11.08275 and token absolute13.71076to15.16027 worsen.',
 'limits':['Only one CPU64 mathematical local candidate was evaluated. This is not proof that every block secant method fails.','Original native BF16 captured inputs and selected original weights are used; derivative/GEMM contractions here are CPU64 algebra, not a native model/FA/FLA forward or production timing claim.','No new needle/RISE/MAS. NumPy verifies public contractions, not the remote private activation/weight products independently.','Do not tune a continuous endpoint weight on these masks or turn exact endpoint closure into a success criterion.']}
path=A/'dt_MH2_decay_block_scout_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'sha256':sha(path)}))
