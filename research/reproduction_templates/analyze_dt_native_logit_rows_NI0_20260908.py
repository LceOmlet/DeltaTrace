"""Independently check actual native row selection, four vectors and original curves."""
import ast,hashlib,json,zipfile,itertools
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';d=A/'snapshot${ARTIFACT_ROOT}/codex_dt_native_logit_rows_NI0_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=json.loads((d/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
with zipfile.ZipFile(d/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name;raw=z.read(name);f=d/name
        if f.exists():assert f.read_bytes()==raw
        else:f.write_bytes(raw)
for name,entry in receipt['files'].items():
    raw=(d/name).read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
r=json.loads((d/'results.json').read_bytes());p=json.loads((d/'protocol.json').read_bytes());assert r['protocol']==p
assert r['status']=='native_logit_rows_complete_ABBA_NI0' and r['DT_calls']==4
assert r['scoring_forwards_entered']==r['scoring_forwards_completed']==84 and r['FT_calls']==r['generation_calls']==0
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
for name,want in p['files_sha256'].items():assert sha((d/name).read_bytes())==want;ast.parse((d/name).read_bytes())
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha((d/name).read_bytes())==want
vectors=np.load(d/'vectors.npz');assert sha((d/'vectors.npz').read_bytes())==r['vectors_sha256']
keep=r['input']['keep'];gold=set(r['input']['gold'])&set(keep);names=[x[0] for x in p['run_order']]
assert list(vectors.files)==names and len(keep)==310 and len(gold)==40
auc=lambda x:float((x.sum()-x[0]/2-x[-1]/2)/(len(x)-1))
metrics={};phases={};endpoints=[]
for name,selected in p['run_order']:
    run=r['runs'][name];w=vectors[name];assert w.shape==(588,) and np.isfinite(w).all()
    assert np.isclose(w.sum(),run['signed_sum'],rtol=0,atol=1e-10)
    assert run['select_output_rows']==selected
    head_rows=248 if selected else 588
    assert run['actual_head_input_shapes']==[[2,head_rows,4096]]
    assert run['actual_output_bytes']==2*head_rows*248320*2
    assert run['selected_predictor_rows']==(list(range(339,587)) if selected else None)
    assert len(run['layers'])==32 and all(v['replay_relative_L2']==0 for v in run['layers'].values())
    assert all(v.get('FA_auxiliary_relative_L2',0)==0 for v in run['layers'].values())
    calls=run['calls'];assert sum(c['kind'].startswith('native_replay_') for c in calls)==32
    assert sum(c['kind'].startswith('finite_decoder_') for c in calls)==32
    assert sum(c['kind'].startswith('public_FA_LSE_') for c in calls)==8
    def seconds(predicate):return sum(c['seconds'] for c in calls if predicate(c['kind']))
    phases[name]={
      'root':seconds(lambda k:k=='native_root_with_CPU_checkpoints'),
      'seed_and_root_logprob':seconds(lambda k:k in ['finite_seed','actual_root_FP32_logprob_diagnostic']),
      'final_norm':seconds(lambda k:k=='finite_final_norm'),
      'native_decoder_replays':seconds(lambda k:k.startswith('native_replay_')),
      'FA_auxiliary':seconds(lambda k:k.startswith('public_FA_LSE_')),
      'finite_FA_decoders':seconds(lambda k:k.startswith('finite_decoder_') and r['runs'][name]['layers'][k.rsplit('_',1)[1]]['block_type']=='full_attention'),
      'finite_GDN_decoders':seconds(lambda k:k.startswith('finite_decoder_') and r['runs'][name]['layers'][k.rsplit('_',1)[1]]['block_type']=='linear_attention'),
    }
    phases[name]['complete_attribution']=run['complete_attribution_seconds_with_diagnostics']
    phases[name]['other_and_observation']=phases[name]['complete_attribution']-sum(v for k,v in phases[name].items() if k!='complete_attribution')
    row=r['curves'][name];wf=w[:340].astype(np.float32);order=row['sorted_keep']
    assert len(order)==310 and set(order)==set(keep) and np.all(np.diff(wf[order])<=0)
    top=order[:31]; assert abs(len(set(top)&gold)/40-run['needle'])<1e-12
    assert np.isclose(float(wf[keep].sum(dtype=np.float32)),row['attr_sum'],rtol=1e-6)
    offset=0;deleted=set();density=[1.]
    for i,entry in enumerate(row['input_receipts']):
        if i:
            group=order[offset:offset+(16 if i<=10 else 15)];offset+=len(group);deleted.update(group)
            density.append(density[-1]-float(wf[group].sum(dtype=np.float32))/row['attr_sum'])
        assert entry['deleted_positions']==sorted(deleted)
    assert offset==310 and np.max(np.abs(density-np.asarray(row['density'])))<1e-6
    score=np.asarray(row['scores'],dtype=np.float64)
    response=np.minimum.accumulate(np.clip((score-score[-1])/abs(score[0]-score[-1]),0,1))
    penalty=np.abs(response-np.asarray(row['density']));corrected=np.clip(response+penalty,0,1)
    corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
    for expected,key in [(response,'normalized_model_response'),(penalty,'alignment_penalty'),(corrected,'corrected_scores')]:assert np.max(np.abs(expected-row[key]))<1e-12
    values=[auc(response),auc(corrected),auc(response+penalty)];assert np.max(np.abs(np.asarray(values)-row['return_metrics']))<1e-12
    metrics[name]={'needle':run['needle'],'RISE':values[0],'MAS':values[1],'G_clean':float(score[0]),'G_EOS':float(score[-1]),
        'negative_eligible_count':int(np.sum(wf[keep]<0)),'top_indices':top}
    endpoints.append((row['input_receipts'][0]['input_sha256'],row['input_receipts'][-1]['input_sha256']))
assert len(set(endpoints))==1 and endpoints[0]==(r['input']['clean_sha256'],r['input']['baseline_sha256'])
pairwise={}
for a,b in itertools.combinations(names,2):
    x,y=vectors[a],vectors[b];lp0=np.asarray(r['runs'][a]['target_logp0']);lp1=np.asarray(r['runs'][b]['target_logp0'])
    pairwise[a+'__'+b]={'vector_equal':bool(np.array_equal(x,y)),'vector_relative_L2':float(np.linalg.norm(y-x)/np.linalg.norm(x)),
        'max_absolute':float(np.max(np.abs(y-x))),'eligible_sign_changes':int(np.sum(np.sign(x[keep])!=np.sign(y[keep]))),
        'target_logp0_equal':bool(np.array_equal(lp0,lp1)),
        'target_logp1_equal':r['runs'][a]['target_logp1']==r['runs'][b]['target_logp1'],
        'original_curves_equal':r['curves'][a]==r['curves'][b]}
warm_A=r['runs']['A1_full'];warm_B=r['runs']['B1_selected']
summary={'status':'native_logit_rows_ABBA_independently_verified','receipt':receipt,'metrics':metrics,'phases':phases,'pairwise':pairwise,
 'all_vectors_identical':all(x['vector_equal'] for x in pairwise.values()),
 'all_original_curves_identical':all(x['original_curves_equal'] for x in pairwise.values()),
 'actual_output_row_reduction_fraction':1-248/588,'output_bytes_saved':warm_A['actual_output_bytes']-warm_B['actual_output_bytes'],
 'single_warm_pair':{'full_seconds':warm_A['complete_attribution_seconds_with_diagnostics'],'selected_seconds':warm_B['complete_attribution_seconds_with_diagnostics'],
    'selected_over_full':warm_B['complete_attribution_seconds_with_diagnostics']/warm_A['complete_attribution_seconds_with_diagnostics'],
    'root_peak_allocated_saved':warm_A['root_peak_allocated']-warm_B['root_peak_allocated'],
    'whole_peak_allocated_full':warm_A['peak_allocated'],'whole_peak_allocated_selected':warm_B['peak_allocated'],
    'reserved_full':warm_A['peak_reserved'],'reserved_selected':warm_B['peak_reserved']},
 'job_seconds':r['job_seconds'],'budget':p['budget'],'model_load_seconds':r['calls'][0]['seconds'],
 'FT_modified':False,'finite_rules_modified':False,'new_samples':0,
 'decision':'Select native target rows by default for the current dense DT runner if all vectors/curves match; retain the full-row option. Structural output/root transient reduction is verified, but this short case shows no material whole-attribution speed or global peak gain. Stop head micro-optimization; use measured larger costs and signed conditional-error diagnosis.',
 'limits':['One existing NI0; same-run exact output equality is observed,not a universal bitwise requirement.',
 'A0/B0 carry unequal initial costs; their ratio is not a speedup. The lateB/lateA warm difference is only one fixed-order pair.',
 'Cumulative vector archive grows by run and is outside the comparable runner time. All archive/housekeeping/needle and model/scoring costs remain in raw records and job cost.',
 'Shared allocator reserved high-water marks persist. Selected rows lower root transient/output bytes,not the observed whole attribution peak.',
 'Original metrics through unchanged scorer and an explicit raw-input formatter view; no stock paper-table or multi-example batching claim.']}
(A/'dt_native_logit_rows_NI0_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','all_vectors_identical','all_original_curves_identical','metrics','phases','output_bytes_saved','single_warm_pair','job_seconds']},ensure_ascii=False))
