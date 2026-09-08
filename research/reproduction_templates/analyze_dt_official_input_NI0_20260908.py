"""Audit saved actual DT endpoints and compare with untouched FT; no model calls."""
import ast, hashlib, json, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent; R=A.parent/'DeltaTrace'
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=json.loads((D/'terminal_receipt.json').read_bytes())
assert receipt['pid']==248547 and not receipt['proc_exists']
for name,meta in receipt['files'].items():
    if (D/name).exists():
        raw=(D/name).read_bytes(); assert sha(raw)==meta['sha256'] and len(raw)==meta['bytes']
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name
        dest=D/name; raw=z.read(name)
        if dest.exists(): assert dest.read_bytes()==raw,name
        else: dest.write_bytes(raw)
p=json.loads((D/'protocol.json').read_bytes());r=json.loads((D/'results.json').read_bytes())
assert sha((D/'protocol.json').read_bytes())=='344d54a0a4e66b7e881affe1f26982cc7ccacb0a0ce11cd49ae856cd4abbb334'
assert (D/'protocol.json').read_bytes()==(A/'dt_official_input_NI0_protocol_20260908.json').read_bytes()
assert r['protocol']==p and r['status']=='DT_content_P1_same_official_NI0_input_completed'
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
for name,digest in p['files_sha256'].items():
    raw=(D/name).read_bytes();assert sha(raw)==digest==r['sources_before'][name];ast.parse(raw)
for name,digest in p['unchanged_finite_runtime_sha256'].items():
    path=R/('core' if name in ['signed_secant_rules.py','compiled_swiglu_secant.py','compiled_finite_rules.py','compiled_logprob_seed.py'] else 'research/runtime')/name
    assert sha(path.read_bytes())==digest,name
ft=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/results.json').read_bytes())
assert sha((A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/results.json').read_bytes())==p['official_FT_results_sha256']
assert all(r['sources_before'][name]==ft['official_sources_before'][name] for name in p['official_package_blob_sha1'])
assert r['input']['clean_sha256']==ft['actual_input']['sha256']=='30ddffc0f75e06ddcb3cc367af440c67b1ca67074702111a56f4463c0b8f7da4'
assert r['input']['paired_shape']==[2,588]
contract=json.loads((A/'snapshot/tmp/qwen35_official_input_contract_20260908.json').read_bytes())
keep=contract['current_keep'];gold=set(contract['current_gold'])&set(keep)
assert len(keep)==310 and len(gold)==40
assert sha((D/'signed_result.npz').read_bytes())==r['artifacts']['signed_result.npz']['sha256']
with np.load(D/'signed_result.npz') as z:
    assert set(z.files)=={'signed','FA_input0','FA_input1','finite'}
    signed=z['signed'];x0=z['FA_input0'];x1=z['FA_input1'];m=z['finite']
    assert signed.shape==(588,) and x0.shape==x1.shape==m.shape==(588,4096)
    assert all(np.isfinite(v).all() for v in [signed,x0,x1,m])
    assert np.flatnonzero(np.any(x1!=x0,axis=1)).tolist()==keep
    reconstructed=(m.astype(np.float64)*(x1.astype(np.float64)-x0.astype(np.float64))).sum(1)
reconstruction_error=float(np.max(np.abs(reconstructed-signed)))
assert reconstruction_error<1e-9
assert np.all(signed[list(set(range(588))-set(keep))]==0)
score=signed[:340].astype(np.float32);positive=np.maximum(score,0)
selected=r['needle']['selected'];hits=sorted(set(selected)&gold)
assert len(set(selected))==31 and set(selected)<=set(keep)
assert hits==r['needle']['hits'] and len(hits)/40==r['needle']['recovery']
cutoff=min(positive[selected]);assert all(positive[j]<=cutoff for j in set(keep)-set(selected))
assert int((score[keep]<0).sum())==r['needle']['negative_eligible_count']
assert int((score[keep]>0).sum())==r['needle']['positive_eligible_count']
assert abs(float(signed.sum())-r['signed_sum'])<1e-9
assert set(r['layers'])==set(map(str,range(32)))
blocks=Counter();replay_loss=0.;finite_loss=0.;max_replay=0.;max_aux=0.
previous=r['seed_effect']
ledger=[]
for i in reversed(range(32)):
    row=r['layers'][str(i)];blocks[row['block_type']]+=1
    assert all(v==1 for v in row['decoder_calls'].values()) and len(row['decoder_calls'])==8
    expected=({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if row['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
    assert row['mixer_calls']==expected
    assert abs(previous-row['root_output_effect'])<1e-8
    rl=row['root_output_effect']-row['replay_output_effect']
    fl=row['replay_output_effect']-row['input_effect']
    replay_loss+=rl;finite_loss+=fl
    max_replay=max(max_replay,row['replay_vs_root']['relative_L2'])
    if 'auxiliary_vs_actual_core' in row:max_aux=max(max_aux,row['auxiliary_vs_actual_core']['relative_L2'])
    ledger.append({'layer':i,'block_type':row['block_type'],'replay_effect_difference':rl,'finite_effect_difference':fl})
    previous=row['input_effect']
assert abs(previous-r['signed_sum'])<1e-9
seed_loss=r['root_effect']-r['seed_effect']
closure_error=abs(seed_loss+replay_loss+finite_loss-r['unassigned_effect'])
assert closure_error<1e-9 and blocks=={'full_attention':8,'linear_attention':24}
assert r['decoder_replays']==r['finite_decoder_calls']==32 and r['auxiliary_FA_calls']==8
assert r['model_loads']==r['finite_seed_calls']==r['original_needle_calls']==1 and r['generation_calls']==0
counts=Counter(x['kind'] for x in r['calls']); costs=defaultdict(float)
for x in r['calls']:
    k=x['kind']
    if k.startswith('actual_decoder'): group='native_decoder_replays'
    elif k.startswith('existing_finite_decoder'):group='finite_decoders'
    elif k.startswith('public_dense_FA'):group='public_FA_auxiliary_LSE'
    else:group=k
    costs[group]+=x['seconds']
assert len(counts)==len(r['calls'])==77
attr_recorded=sum(v for k,v in costs.items() if k not in ['complete_model_load','native_eager_B1_reference'])
costs['other_in_full_attribution']=r['complete_attribution_seconds_with_diagnostics']-attr_recorded
assert costs['other_in_full_attribution']>=0
costs['complete_attribution_with_diagnostics']=r['complete_attribution_seconds_with_diagnostics']
costs['job_before_bundle']=r['job_seconds']
costs['other_job_setup_and_final_exports']=r['job_seconds']-costs['complete_model_load']-costs['native_eager_B1_reference']-costs['complete_attribution_with_diagnostics']
finite_rows=[x for x in r['calls'] if x['kind'].startswith('existing_finite_decoder')]
summary={'status':r['status'],'all_jobs_terminal':True,'input':r['input'],
 'protocol_sha256':sha((D/'protocol.json').read_bytes()),'results_sha256':sha((D/'results.json').read_bytes()),
 'receipt':receipt,'unchanged_finite_runtime_count':len(p['unchanged_finite_runtime_sha256']),
 'official_FT_package_unchanged_count':len(p['official_package_blob_sha1']),
 'needle':{**r['needle'],'eligible_count':310,'eligible_gold_count':40,'selected_count':31,'cutoff_ties':int((positive[keep]==cutoff).sum())},
 'official_FT_needle':ft['needle'],'backend_diagnostic':r['backend_diagnostic'],
 'finite_audit':{'root_effect':r['root_effect'],'seed_effect':r['seed_effect'],'signed_sum':r['signed_sum'],
  'unassigned_effect':r['unassigned_effect'],'relative_residual':r['relative_residual'],
  'embedding_product_max_abs_error':reconstruction_error,'seed_effect_difference':seed_loss,
  'total_replay_effect_difference':replay_loss,'total_finite_decoder_effect_difference':finite_loss,
  'ledger_closure_error':closure_error,'max_replay_relative_L2':max_replay,'max_auxiliary_FA_relative_L2':max_aux,'layer_ledger':ledger},
 'calls':{k:r[k] for k in ['model_loads','root_calls','decoder_replays','finite_decoder_calls','auxiliary_FA_calls','finite_seed_calls','original_needle_calls','generation_calls']},
 'cost_seconds':dict(costs),'largest_finite_calls':sorted(finite_rows,key=lambda x:x['seconds'],reverse=True)[:5],
 'memory_bytes':{k:r[k] for k in ['peak_allocated','peak_reserved']},
 'FT_cost_seconds':ft['timings'],'FT_memory_bytes':{k:ft[k] for k in ['peak_allocated','peak_reserved']},
 'limits':['One historically used NI0 example, not independent confirmation.',
  'Same checkpoint/input/fixed response/eligible set; DT FA+B2 versus official FT eager+B1. Backend and batch differences measured, not eliminated.',
  'DT signed whole-response logprob finite propagation and author FT representation/recursive target retain their own definitions.',
  'Cold compilation, checkpoint IO and passive diagnostics are included; no matched warmed speed conclusion.',
  'Endpoint B2 is one attribution example, not demonstrated multi-example minibatching.',
  'Finite conservation and negative coefficients do not prove individual deletion signs; original RISE/MAS and deletion-sign validation pending.'],
 'next':'Use unchanged original skip-token faithfulness evaluator for frozen NI0 vectors first, after verifying its actual tokenizer/model input. Do not alter FT or tune on deleted scores.'}
(A/'dt_official_input_NI0_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
vectors={'source_npz_sha256':r['artifacts']['signed_result.npz']['sha256'],'input_ids_sha256':r['input']['clean_sha256'],
 'description':'Signed numeric attribution only; no text/token IDs, no rescaling, positive clamp only inside original recovery metric.',
 'signed':signed.tolist()}
(A/'dt_official_input_NI0_score_vector_20260908.json').write_text(json.dumps(vectors,separators=(',',':')),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','needle','cost_seconds','largest_finite_calls','memory_bytes','limits']},ensure_ascii=False))
