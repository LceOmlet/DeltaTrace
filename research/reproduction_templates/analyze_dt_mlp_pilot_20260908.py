"""Independently audit paired MLP result vectors, source lineage and full ledger."""
import ast,json,hashlib,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();jobs=[]
for version in ['v1','v2']:
 d=A/('snapshot${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_'+version)
 receipt=json.loads((d/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
 with zipfile.ZipFile(d/'review_bundle.zip') as z:
  assert z.testzip() is None
  for name in z.namelist():
   assert Path(name).name==name;f=d/name;raw=z.read(name)
   if f.exists():assert f.read_bytes()==raw
   else:f.write_bytes(raw)
 for name,item in receipt['files'].items():
  raw=(d/name).read_bytes();assert sha(raw)==item['sha256'] and len(raw)==item['bytes']
 r=json.loads((d/'results.json').read_bytes());p=r['protocol'];assert p==json.loads((d/'protocol.json').read_bytes())
 for name,want in p['files_sha256'].items():assert sha((d/name).read_bytes())==want;ast.parse((d/name).read_bytes())
 assert len(r['layers'])==32 and r['decoder_replays']==32 and r['finite_decoder_calls']=={'symmetric':32,'content1':32}
 assert r['FA_auxiliary_calls']==8 and r['original_needle_calls']==2 and r['model_loads']==1
 jobs.append({'version':version,'status':r['status'],'job_seconds':r['job_seconds'],'receipt':receipt,'error':r.get('error'),
  'model_loads':r['model_loads'],'root_forwards':r['root_forwards'],'decoder_replays':r['decoder_replays'],
  'finite_decoder_calls':r['finite_decoder_calls'],'methods':r['methods'],'full_vectors_saved':(d/'signed_vectors.npz').exists()})
 if version=='v1':assert r['status']=='failed' and not (d/'signed_vectors.npz').exists()
 else:last=d;actual=r
assert actual['status']=='paired_DT_MLP_content1_NI0_completed'
assert actual['sources_before']==actual['sources_after'] and actual['weight_stats_before']==actual['weight_stats_after']
assert sha((last/'signed_vectors.npz').read_bytes())==actual['artifacts']['signed_vectors.npz']['sha256']
z=np.load(last/'signed_vectors.npz');old=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1/signed_result.npz')['signed']
contract=json.loads((A/'snapshot/tmp/qwen35_official_input_contract_20260908.json').read_bytes());keep=contract['current_keep'];gold=set(contract['current_gold'])&set(keep)
assert set(z.files)=={'symmetric','content1'} and len(gold)==40
checks={};costs=defaultdict(float)
for method in z.files:
 v=z[method];row=actual['methods'][method];score=v[:340].astype(np.float32);positive=np.maximum(score,0)
 assert v.shape==(588,) and np.isfinite(v).all();assert np.all(v[list(set(range(588))-set(keep))]==0)
 selected=row['selected'];assert len(set(selected))==31 and set(selected)<=set(keep)
 hits=set(selected)&gold;assert sorted(hits)==row['hits'] and len(hits)/40==row['needle']
 cutoff=min(positive[selected]);assert all(positive[j]<=cutoff for j in set(keep)-set(selected))
 assert abs(float(v.sum())-row['signed_sum'])<1e-9
 previous=actual['seed_effect'];replay=finite=0.;ledger=[]
 for i in reversed(range(32)):
  lr=actual['layers'][str(i)];part=lr['methods'][method]
  assert abs(previous-part['root_output_effect'])<1e-9
  a=part['root_output_effect']-part['replay_output_effect'];b=part['replay_output_effect']-part['input_effect']
  replay+=a;finite+=b;previous=part['input_effect'];ledger.append({'layer':i,'replay_effect_difference':a,'finite_effect_difference':b})
 assert abs(previous-row['signed_sum'])<1e-9
 expected=actual['root_effect']-actual['seed_effect']+replay+finite
 closure=abs(expected-(actual['root_effect']-row['signed_sum']));assert closure<1e-9
 checks[method]={'needle':row['needle'],'ledger_closure_error':closure,'total_replay_effect_difference':replay,'total_finite_effect_difference':finite,
  'relative_residual':row['relative_residual'],'positive_count':int((score[keep]>0).sum()),'negative_count':int((score[keep]<0).sum()),'layer_ledger':ledger}
for c in actual['calls']:
 k=c['kind'];group=('symmetric_finite' if k.startswith('symmetric_finite') else 'content1_finite' if k.startswith('content1_finite') else 'native_replays' if k.startswith('actual_decoder') else 'public_FA_LSE' if k.startswith('public_FA') else k)
 costs[group]+=c['seconds']
comparison=actual['control_comparison'];assert np.isclose(comparison['score_relative_L2'],np.linalg.norm(z['symmetric']-old)/np.linalg.norm(old))
assert comparison['eligible_sign_changes']==int(np.sum(np.sign(z['symmetric'][keep])!=np.sign(old[keep])))
baseline_hits=set(actual['methods']['symmetric']['hits']);new_hits=set(actual['methods']['content1']['hits'])
summary={'status':'paired_MLP_pilot_independently_audited','jobs':jobs,'all_jobs_terminal':True,
 'candidate_sha256':actual['protocol']['files_sha256']['mlp_content1_finite.py'],'input_sha256':actual['input']['clean_sha256'],
 'methods':actual['methods'],'checks':checks,'control_comparison':comparison,'symmetric_vs_saved_score_max_abs':actual['symmetric_vs_saved_score_max_abs'],
 'max_native_replay_relative_L2':max(v['replay_vs_saved_root']['relative_L2'] for v in actual['layers'].values()),
 'needle_lost_gold_positions':sorted(baseline_hits-new_hits),'needle_gained_gold_positions':sorted(new_hits-baseline_hits),
 'paired_cost_seconds':dict(costs),'paired_propagation_seconds':actual['paired_propagation_seconds_with_diagnostics'],
 'peak_allocated':actual['peak_allocated'],'peak_reserved':actual['peak_reserved'],
 'actual_total_budget':{'model_loads':2,'new_root_forwards':0,'native_decoder_replays':64,'finite_decoder_calls':128,'public_FA_auxiliary_calls':16,'original_needle_calls':4,'FT_runs':0,'generation':0,'deletion':0},
 'preliminary_decision':'Candidate needle decreases37.5% to32.5% on NI0; retain frozen symmetric default. Original NI0 RISE/MAS will determine whether any deletion-fidelity compensation exists; do not infer broad inferiority from one case.',
 'limits':['V1 remains a failed driver run with missing full vectors; v2 is a separately frozen evidence-saving recovery, not retroactive passing.',
  'Default native numerical drift is recorded, not treated as bitwise equivalence; one near-zero sign changes in the symmetric control versus saved baseline.',
  'Shared cached actual roots and paired layer replays do not establish full or fair steady attribution speed; original endpoint computation costs remain in source lineage.']}
(A/'dt_mlp_pilot_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
vectors={'source_npz_sha256':sha((last/'signed_vectors.npz').read_bytes()),'input_sha256':summary['input_sha256'],'arrays':{k:z[k].tolist() for k in z.files}}
(A/'dt_mlp_pilot_score_vectors_20260908.json').write_text(json.dumps(vectors,separators=(',',':')),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','methods','control_comparison','needle_lost_gold_positions','needle_gained_gold_positions','paired_cost_seconds','paired_propagation_seconds','actual_total_budget']},ensure_ascii=False))
