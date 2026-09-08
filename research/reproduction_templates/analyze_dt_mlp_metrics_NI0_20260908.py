"""Recompute original metrics and deletion paths from all seven saved native curves."""
import ast,json,hashlib,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp_original_metrics_NI0_20260908_v1';sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
with zipfile.ZipFile(D/'review_bundle.zip') as z:
 assert z.testzip() is None
 for name in z.namelist():
  assert Path(name).name==name;f=D/name;raw=z.read(name)
  if f.exists():assert f.read_bytes()==raw
  else:f.write_bytes(raw)
for name,info in receipt['files'].items():
 raw=(D/name).read_bytes();assert sha(raw)==info['sha256'] and len(raw)==info['bytes']
r=json.loads((D/'results.json').read_bytes());p=r['protocol'];assert p==json.loads((D/'protocol.json').read_bytes())
assert r['status']=='original_RISE_MAS_on_fixed_NI0_input_completed'
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
for name,want in p['files_sha256'].items():assert sha((D/name).read_bytes())==want;ast.parse((D/name).read_bytes())
assert r['model_forwards_started']==r['model_forwards_completed']==147
assert r['model_loads']==1 and r['FT_runs']==r['attribution_calls']==r['generation_calls']==0
contract=json.loads((A/'snapshot/tmp/qwen35_official_input_contract_20260908.json').read_bytes());keep=contract['current_keep'];keep_set=set(keep)
ft=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/official_outputs.npz')['scores']
base=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1/signed_result.npz')['signed']
paired=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v2/signed_vectors.npz')
vectors={**{f'FT{i}':ft[i] for i in range(4)},'DT_frozen':base[:340],'DT_sym_control':paired['symmetric'][:340],'DT_MLP_content1':paired['content1'][:340]}
auc=lambda x:float((np.sum(x)-x[0]/2-x[-1]/2)/(len(x)-1))
metrics={};score_errors=[];endpoint_hashes=[]
for name,row in r['curves'].items():
 w=np.asarray(vectors[name],dtype=np.float32);assert sha(w.tobytes())==r['scores_sha256'][name]
 order=row['sorted_keep'];assert len(order)==310 and set(order)==keep_set
 assert np.all(np.diff(w[order])<=0)
 arrays={k:np.asarray(row[k],dtype=np.float64) for k in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']}
 assert all(v.shape==(21,) and np.isfinite(v).all() for v in arrays.values())
 density=[1.];offset=0;deleted=set();attr_sum=row['attr_sum']
 assert np.isclose(attr_sum,float(w[keep].sum(dtype=np.float32)),rtol=1e-6)
 for i,entry in enumerate(row['input_receipts']):
  if i:
   size=16 if i<=10 else 15;group=order[offset:offset+size];offset+=size;deleted.update(group)
   density.append(density[-1]-float(w[group].sum(dtype=np.float32))/attr_sum if attr_sum>0 else 1-i/20)
  assert entry['deleted_positions']==sorted(deleted)
  if i==0:assert entry['input_sha256']==p['input_sha256']
 assert offset==310 and np.max(np.abs(np.asarray(density)-arrays['density']))<1e-6
 score=arrays['scores'];response=np.minimum.accumulate(np.clip((score-score[-1])/abs(score[0]-score[-1]),0,1))
 penalty=np.abs(response-arrays['density']);corrected=np.clip(response+penalty,0,1)
 corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
 if np.isnan(corrected).any():corrected=np.linspace(1,0,21)
 for expected,key in [(response,'normalized_model_response'),(penalty,'alignment_penalty'),(corrected,'corrected_scores')]:assert np.max(np.abs(expected-arrays[key]))<1e-12
 computed=[auc(response),auc(corrected),auc(response+penalty)]
 assert np.max(np.abs(np.array(computed)-row['return_metrics']))<1e-12
 outside=np.maximum(-arrays['density'],0)+np.maximum(arrays['density']-1,0)
 metrics[name]={'RISE':computed[0],'MAS':computed[1],'RISE_plus_AP':computed[2],
  'alignment_penalty_auc':auc(penalty),'density_outside_unit_interval_auc':auc(outside),
  'density_min':float(arrays['density'].min()),'density_max':float(arrays['density'].max()),'density_increases':row['density_increases'],
  'positive_density_excess_auc':auc(np.maximum(arrays['density']-response,0)),
  'response_excess_over_density_auc':auc(np.maximum(response-arrays['density'],0)),
  'clean_logprob':float(score[0]),'all_EOS_logprob':float(score[-1]),'scoring_seconds':row['seconds'],
  'peak_allocated':row['peak_allocated'],'peak_reserved':row['peak_reserved']}
 endpoint_hashes.append((row['input_receipts'][0]['input_sha256'],row['input_receipts'][-1]['input_sha256']))
assert len(set(endpoint_hashes))==1
best_ft={key:min(metrics[f'FT{i}'][key] for i in range(4)) for key in ['RISE','MAS']}
candidate_delta={key:metrics['DT_MLP_content1'][key]-metrics['DT_sym_control'][key] for key in ['RISE','MAS']}
assert all(x>0 for x in candidate_delta.values())
baseline_delta={key:metrics['DT_frozen'][key]-best_ft[key] for key in ['RISE','MAS']}
summary={'status':'original_NI0_metrics_independently_recomputed','receipt':receipt,'metrics':metrics,
 'input_adapter':r['input_adapter'],'official_FT_source_modified':False,'original_metric_and_scorer_sources_unchanged':True,
 'all_curve_endpoints_have_same_input_hashes':True,'all_clean_logprobs':sorted(set(m['clean_logprob'] for m in metrics.values())),
 'all_EOS_logprobs':sorted(set(m['all_EOS_logprob'] for m in metrics.values())),
 'best_FT':best_ft,'candidate_minus_paired_symmetric':candidate_delta,'frozen_DT_minus_best_FT':baseline_delta,
 'actual_budget':{'model_loads':r['model_loads'],'native_B1_forwards':147,'curves':7,'steps_per_curve':20,'FT_runs':0,'attribution_runs':0,'generation':0,'new_samples':0},
 'job_seconds':r['job_seconds'],'model_load_seconds':r['calls'][0]['seconds'],
 'decision':'Do not promote MLPcontent1. On soleNI0 it worsens needle,RISE andMAS against paired symmetricDT. Stop candidate expansion and retain frozen currentDT; proceed to native logits_to_keep efficiency work. FrozenDT RISE leads bestFT onNI0, but MAS remains worse. No broad/independent victory claim.',
 'limits':['Original RISE/MAS function and actual original scoring execute unchanged, with an explicit external formatter view to retain the actual588-token attribution input. Stock evaluator would use605tokens. This is not unmodified run_exp or paper-table reproduction.',
  'One historically used NI0 example; no Morehop or held-out inference. DefaultBF16 native scoring retained.',
  'MAS density and alignment diagnosed offline only; no score normalization/clipping or metric modification was used.']}
(A/'dt_mlp_metrics_NI0_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','metrics','best_FT','candidate_minus_paired_symmetric','frozen_DT_minus_best_FT','all_clean_logprobs','all_EOS_logprobs','job_seconds','decision']},ensure_ascii=False))
