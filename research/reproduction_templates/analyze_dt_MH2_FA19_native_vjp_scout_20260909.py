"""NumPy audit of actual public FA/autograd conditional contrasts; no native work."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_FA19_native_vjp_scout_20260909_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum())}
p,r=read(D/'protocol.json'),read(D/'results.json');assert r['protocol']==p and r['status']=='MH2_FA19_one_public_FA_one_native_VJP_scout_complete'
assert sha(D/'protocol.json')=='c63ba58041536b1a9450593b07abaa75ea822f4b7772af3ce57eac39644c66bc'
assert p['files_sha256']['study.py']=='7bf7f520949927b8268576528392c9d712ec2798e22ce94437cd442e436de026'
receipt=read(D/'terminal_receipt.json');assert receipt['proc_exists'] is False
for n,x in receipt['files'].items():assert sha(D/n)==x['sha256'] and (D/n).stat().st_size==x['bytes']
for n,h in p['files_sha256'].items():assert sha(D/n)==h
with zipfile.ZipFile(D/'review_bundle.zip') as z:
 for n in z.namelist():assert not n.endswith('.pt') and z.read(n)==(D/n).read_bytes()
assert r['native_FA_entered']==r['native_FA_returned']==r['native_autograd_entered']==r['native_autograd_returned']==1
assert all(r[k]==0 for k in ['model_calls','DT_calls','scorer_calls','FT_calls','generation_calls'])
for item in p['protected_sources']:assert sha(A/'snapshot'/item['path'].lstrip('/'))==item['sha256']
source=read(A/'snapshot'/p['source']['results'].lstrip('/'));boundary=read(A/'snapshot'/p['boundary_results'].lstrip('/'));info=source['input'];T,P=info['total_length'],info['prompt_length'];keep=set(info['keep'])
v=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['artifacts']['vectors.npz']['sha256'];keys=set()
for step,row in r['points'].items():
 fields={n:v[step+'_'+n] for n in row['fields']};keys.update(step+'_'+n for n in fields)
 deleted=keep if step=='B2' else set(boundary['cases']['morehopqa_2']['points'][step]['input_receipt']['deleted_positions'])
 groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
 for n,x in fields.items():
  assert x.shape==(1,T) and x.dtype==np.float64 and np.isfinite(x).all()
  for k,a in stats(x).items():close(a,row['fields'][n][k])
  for g,ix in groups.items():
   assert row['groups'][g]['count']==len(ix)
   for k,a in stats(x[0,ix]).items():close(a,row['groups'][g]['fields'][n][k])
 for method in ['current','native_vjp']:
  close(fields[method+'_qk']+fields[method+'_value'],fields[method+'_prediction'])
  close(fields[method+'_prediction']-fields['actual_effect'],fields[method+'_error'])
 close(fields['current_error'].sum(),source['layers']['19']['ledgers'][step]['internal']['terms']['finite_FA_core_including_seed_cast'])
assert set(v.files)==keys
assert all(r['points'][s]['fields']['native_vjp_error']['net']>r['points'][s]['fields']['current_error']['net']>0 for s in ['3','10'])
out={'status':'MH2_native_FA_VJP_scout_independent_CPU_audit_passed_candidate_rejected','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),
 'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'points':r['points'],'native_output_drift':r['native_output_drift'],'job_seconds':r['seconds'],
 'actual_budget':p['budget'],'native_FA_seconds':r['native_FA_seconds'],'native_autograd_seconds':r['native_autograd_seconds'],'private_coefficients':r['artifacts']['native_vjp_coefficients_private.pt'],
 'decision':'Reject native input-FA VJP as this FA19 local repair. Early/mid core errors11.59663/14.36573 become60.72106/55.73312. Value path is nearly unchanged; QK prediction grows sharply. No full quality expansion or runtime VJP integration. Do not claim ordinary native backward is erroneous.',
 'limits':['Actual standard public BF16 FA forward and autograd were used. The stored seed retains its normal BF16 conversion. No ordinary attention/backward shadow implementation.',
 'One FA core with fixed actual upstream and perturbations; not a full input VJP or a proof about every gradient method. No new needle/RISE/MAS or production timing claim.',
 'Conditional core and endpoint discrepancy retained; group terms are contraction positions, not separate source causal effects. NumPy verifies public identities, not private gradient dot products independently.']}
path=A/'dt_MH2_FA19_native_vjp_scout_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'sha256':sha(path)}))
