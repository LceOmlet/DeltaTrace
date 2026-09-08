"""Independent CPU audit of saved MH0 FA19 nine-term and transfer ledgers."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_internal_20260909_v1'
started=time.perf_counter();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(x,y,label='',tol=1e-7):
    err=float(np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0));assert err<tol,(label,err)
    return err
def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum())}
p,r=read(D/'protocol.json'),read(D/'results.json')
assert p==r['protocol']==read(A/'dt_MH0_FA19_internal_protocol_20260909.json')
assert sha(D/'protocol.json')=='fd081e99b5125442d7abd0312844c8ad7fa9f05a20ae11378dafc0f0ab5edb8f'
assert p['files_sha256']['study.py']=='9cf0680bb02694a8ab5ecb33063f22da4e3a5308c13cf9a98c82592b2cfb2f76'
assert r['status']=='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete'
receipt=read(D/'terminal_receipt.json');assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():
    assert sha(D/name)==(item if isinstance(item,str) else item['sha256']),name
    if isinstance(item,dict) and 'bytes' in item:assert (D/name).stat().st_size==item['bytes']
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    assert not any(name.endswith('.pt') for name in archive.namelist())
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
assert r['model_loads']==r['whole_model_entered']==r['whole_model_returned']==1
assert r['layer_replays_entered']==r['layer_replays_returned']==5
assert r['finite_layer_entered']==r['finite_layer_returned']==r['auxiliary_FA_entered']==r['auxiliary_FA_returned']==1
assert r['finite_FA_counts']=={'entered':1,'returned':1}
assert all(r[k]==0 for k in ['DT_calls','scorer_calls','FT_calls','generation_calls'])
assert all(call['status']=='returned' and call['seconds']>=0 for call in r['calls'])
assert r['native_call_accounting']=={'FA_from_returned_whole_forwards':8,'GDN_FLA_and_conv_each_from_returned_whole_forwards':24,
    'FA_from_returned_single_layer_replays':5,'nonreturned_whole_or_layer_internal_calls':0,'nonreturned_finite_native_stages':0}
source=p['source_boundary'];S=(A/'snapshot'/source['results_path'].lstrip('/')).parent
for name in ['results','protocol','vectors']:assert sha(S/(name+('.npz' if name=='vectors' else '.json')))==source[name+'_sha256']
prior=read(S/'results.json');pv=np.load(S/'vectors.npz',allow_pickle=False);v=np.load(D/'vectors.npz',allow_pickle=False)
assert r['input']==p['input']==prior['cases']['morehopqa_0']['input'];T,P=r['input']['total_length'],r['input']['prompt_length'];keep=set(r['input']['keep'])
assert set(r['points'])==set(p['capture_steps'])=={'B2','0','3','10','20'}
k=r['kwargs_capture'];assert set(p['required_decoder_kwargs']).issubset(k['actual_keys'])
assert k['position_ids_shape']==[2,T] and k['cos_sin_shape'][:2]==[2,T] and k['actual_FA_mask'] is None and k['paired_position_rows_equal']
fa_args=[]
for name,row in r['points'].items():
    assert row['decoder_calls']=={key:1 for key in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
    assert row['mixer_calls']=={'module':1,'interface':1,'native_varlen':0,'native_dense':1}
    cache=row['initial_cache'];assert cache['provided']==(name!='B2') and cache['selected_layer_initial_length']==0
    assert cache['selected_layer_final_length']==(None if name=='B2' else T)
    fa_args.append(row['native_dense_arguments'])
assert all(args==fa_args[0] for args in fa_args) and fa_args[0]['causal'] and fa_args[0]['dropout_p']==0
assert r['FA_activity']['GQA_input_expansion'] is False
out={'status':'MH0_FA19_internal_independent_CPU_audit_passed','next':None,'analyzer_sha256':sha(Path(__file__)),
    'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),'vectors_sha256':sha(D/'vectors.npz'),
    'source_boundary':source,'input':{k:z for k,z in r['input'].items() if k!='keep'},
    'actual_budget':{'model_loads':1,'native_B2_kwargs_whole_forwards':1,'selected_native_decoder_replays':5,
        'selected_finite_decoder_calls':1,'native_FA_in_whole_forward':8,'native_GDN_FLA_and_conv_each_in_whole_forward':24,
        'native_FA_in_single_layer_replays':5,'public_FA_LSE_auxiliary':1,'finite_FA':1,'finite_FLA':0,
        'DT':0,'scorer':0,'FT':0,'generation':0,'audit_model_calls':0,'job_seconds':r['seconds'],
        'GPU_peak_allocated_full_job':r['GPU_peak_allocated_full_job'],'GPU_peak_reserved_full_job':r['GPU_peak_reserved_full_job'],
        'failed_entered_minus_returned':0,'scope':'Actual diagnostic job, including copies, CPU contractions and private save, not production timing.'},
    'm19_replay_drift':r['m19_replay_drift_report_only'],'native_output_replay_drift':{s:q['output_replay_drift'] for s,q in r['points'].items()},
    'kwargs_capture':r['kwargs_capture'],'B2_FA_auxiliary_drift':r['B2_public_FA_auxiliary_output_drift'],
    'native_dense_arguments':fa_args[0],'private_artifact':r['private_artifact'],'points':{},
    'proof_scope':'Independent NumPy audit of all saved signed token contractions and identities; remote private coefficient/activation tensors are not locally rehashed or independently dot-product recomputed. Existing helper/source identities and actual-call receipts are verified.'}
allkeys=set()
for step in ['3','10','20','B2']:
    row=r['conditional_ledgers'][step];ledger=row['replayed_9term_ledger'];assert ledger['sign_convention']=='prediction_minus_actual' and len(ledger['terms'])==9
    transfers=row['transfer_terms'];assert set(transfers)=={'saved_minus_replayed_m19','input_replay_transfer','output_replay_transfer'}
    deleted=keep if step=='B2' else set(prior['cases']['morehopqa_0']['points'][step]['input_receipt']['deleted_positions'])
    groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    arrays={key:v[step+'_'+key] for key in row['coordinate_groups']};allkeys.update(step+'_'+key for key in arrays)
    assert set(arrays)==set(ledger['terms'])|set(transfers)|{'saved_boundary_error'}
    for name,x in arrays.items():
        assert x.shape==(1,T) and x.dtype==np.float64 and np.isfinite(x).all()
        saved=row['coordinate_groups'][name]
        for k,value in stats(x).items():close(value,saved['total'][k])
        for group,indices in groups.items():
            assert saved['groups'][group]['count']==len(indices)
            for k,value in stats(x[:,indices]).items():close(value,saved['groups'][group][k])
    for name,value in ledger['terms'].items():close(arrays[name].sum(),value,name)
    for name,values in transfers.items():
        for k,value in stats(arrays[name]).items():close(value,values[k],name)
    measured=arrays['saved_boundary_error'];reconstruction=sum(arrays[name] for name in ledger['terms'])+sum(arrays[name] for name in transfers)
    maximum=close(reconstruction,measured,step+' token total')
    close(sum(ledger['terms'].values()),ledger['input_contraction']-ledger['output_contraction'])
    close(ledger['prediction_minus_actual'],sum(ledger['terms'].values()));close(ledger['actual_minus_predicted'],-ledger['prediction_minus_actual'])
    if step=='B2':expected=prior['runs'][0]['B2_endpoint_boundary_contractions']['19']-prior['runs'][0]['B2_endpoint_boundary_contractions']['20']
    else:
        expected=-prior['cases']['morehopqa_0']['decomposition'][step]['decoder_errors']['19']
        close(measured[0],-pv['step'+step+'_decoder_19_actual_minus_predicted'],step+' actual source token boundary')
    close(measured.sum(),expected,step+' source scalar')
    terms=ledger['terms'];out['points'][step]={'saved_boundary_error':stats(measured),'replayed_9term_ledger':ledger,
        'transfer_terms':transfers,'transfer_total':sum(q['net'] for q in transfers.values()),
        'token_closure_max_absolute':maximum,'positive_terms_ranked':sorted([(k,z) for k,z in terms.items() if z>0],key=lambda pair:-pair[1]),
        'negative_terms_ranked':sorted([(k,z) for k,z in terms.items() if z<0],key=lambda pair:pair[1]),
        'coordinate_groups':row['coordinate_groups']}
assert set(v.files)==allkeys
mid=out['points']['10']['replayed_9term_ledger']['terms'];early=out['points']['3']['replayed_9term_ledger']['terms']
core='finite_FA_core_including_seed_cast'
if max(mid,key=lambda k:abs(mid[k]))==core:
    out['next']={'status':'native_FA_hybrid_quantification_preparation_authorized_by_root',
        'observed_basis':'FA core is largest absolute mid-step nine-term contribution; retain early,mid and allEOS compensation. This does not identify routing versus value reference or precision by itself.',
        'reuse':'Existing successful dt_fa19_native_hybrid_20260908_v2.py eleven-public-FA-call schedule, adapting only current stored schema and early3 replacing early1.',
        'budget':{'native_FA_calls':11,'model':0,'DT':0,'scorer':0,'FT':0,'generation':0,'backward':0},
        'quantification':'Four native output replays; one shared clean-routing-at-B2-V0 hybrid plus two hybrids each for early3/mid10/allEOS20. Decompose qk prediction minus R0, value prediction minus VCVA, and R0 minus RA; retain actual captured outputs, native replay drift, BF16 seed-cast boundary and exact GQA folding.',
        'limit':'No candidate replay or new kernel. If routing dominates, only then consider the existing compiled QK/softmax diagnostic library under a separate bound.'}
else:out['next']={'status':'root_review_observed_noncore_dominant_terms_before_any_further_call'}
out['observed_primary_terms']={s:{k:out['points'][s]['replayed_9term_ledger']['terms'][k] for k in [core,'attention_output_projection_and_sigmoid_gate','MLP_combined','input_RMSNorm']} for s in ['3','10','20','B2']}
out['interpretation']='Nine terms and three transfer terms localize current saved-boundary disagreement. Negative output-gate, MLP or norm terms may compensate a positive core term and are not automatically faults. A large core does not establish an implementation bug, identify softmax versus PV, or predict full-propagation MAS benefit.'
out['CPU_audit_seconds']=time.perf_counter()-started
path=A/'dt_MH0_FA19_internal_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'summary':str(path),'sha256':sha(path),'primary_terms':out['observed_primary_terms'],'next':out['next'],'seconds':out['CPU_audit_seconds']}))
