"""Independent NumPy/source audit of the captured NI FLA reference ledger."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_NI_fla_reference_mismatch_20260908_v1'
started=time.perf_counter()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_bytes())
p=read(D/'protocol.json');r=read(D/'results.json')
receipt=read(D/'terminal_receipt.json') if (D/'terminal_receipt.json').exists() else None
assert p==r['protocol']==read(A/'dt_NI_fla_reference_mismatch_protocol_20260908.json')
assert sha(D/'protocol.json')=='87bf80252e58097243523bd5b0a22519fd0f1eaac14feeafb51812e03283ee9b'
assert r['status']=='NI_FLA_reference_content_native_capture_and_CPU_audit_complete'
if receipt is not None:
    assert receipt.get('proc_exists') is False
    for name,v in receipt['files'].items():
        assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name
for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes(),name
assert sha(D/'vectors.npz')==r['vectors_sha256']
source_path=A/'snapshot'/p['source_results_path'].lstrip('/')
boundary_path=A/'snapshot'/p['boundary_results_path'].lstrip('/')
assert sha(source_path)==p['source_results_sha256'] and sha(boundary_path)==p['boundary_results_sha256']
source=read(source_path);boundary=read(boundary_path)
assert p['input']==source['input']==boundary['input']
assert p['source_private_sha256']==source['private_artifact']['sha256']
assert p['source_private_bytes']==source['private_artifact']['bytes']
assert r['native_input_adjoints_entered']==r['native_input_adjoints_returned']==1
assert r['mixed_entered']==r['mixed_returned']==1
assert all(r[k]==0 for k in ('model_loads','model_forwards','scorer_calls','FT_calls','FA_calls'))
assert all(v['status']=='returned' for v in r['calls'])
assert [v['kind'] for v in r['calls']]==['load_saved_actual_private_CPU','existing_native_input_adjoints_two_stages',
    'existing_compiled_mixed_with_retained_outputs','CPU64_original_chunk_reference_algebra']
cpu=r['CPU_audit'];N=(p['input']['total_length']+63)//64
assert cpu['counts']=={'mixed_chunk_CPU64_calls':4*N,'extra_key_read_CPU_matmuls':12*N,
    'A_transpose_du_recomputations':0,'native_calls':0,'model_calls':0,'GPU_calls':0}
assert 4*N==p['budget']['CPU_reference_mixed_chunk_calls']==76

def close(x,y):
    value=float(np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0))
    assert value<1e-7,value
    return value

def stats(x):
    return {'net':float(x.sum()),'positive':float(np.maximum(x,0).sum()),
        'negative':float(np.minimum(x,0).sum()),'absolute':float(np.abs(x).sum())}

branches=('q','k','v','beta','g')
numerical_names=[prefix+k for prefix in ('saved_to_replayed_GPU_','replayed_GPU_to_CPU01_') for k in branches]
reference_names=['query_reference_content','key_write_reference_content','key_read_reference_content',
    'beta_reference_correction_content','v_reference_difference','decay_reference_content']
exp_name='exp_secant_conditional_curvature'
remainder_name='matched_pair_native_and_fixed_adjoint_closure'
transfer_name='B1clean_minus_B2input_saved_coefficient_transfer'
term_names=numerical_names+reference_names+[exp_name,remainder_name,transfer_name]
aux_names=['saved_error','B2input_minus_A_actual_output','MA1_primitive_prediction','closure_residual']
T=p['input']['total_length'];P=p['input']['prompt_length'];keep=set(p['input']['keep'])
H=r['actual_input_layout']['q']['shape'][2]
assert r['actual_input_layout']['q']['shape']==[2,T,H,128]
z=np.load(D/'vectors.npz',allow_pickle=False)
assert set(z.files)=={step+'_'+name for step in ('1','10','20','B2') for name in term_names+aux_names}
out={'status':'independent_NI_FLA_reference_ledger_audit_passed' if receipt is not None else 'source_and_NPZ_passed_terminal_receipt_pending',
    'terminal_receipt_verified':receipt is not None,'analyzer_sha256':sha(Path(__file__)),
    'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),'vectors_sha256':sha(D/'vectors.npz'),
    'source_results_sha256':sha(source_path),'source_private_artifact':source['private_artifact'],
    'adjoint_capture_artifact':r['adjoint_capture_artifact'],'input':p['input'],'seconds':r['seconds'],
    'actual_calls':r['calls'],'counts':{k:r[k] for k in ('native_input_adjoints_entered','native_input_adjoints_returned',
        'mixed_entered','mixed_returned','model_loads','model_forwards','scorer_calls','FA_calls','FT_calls')},
    'CPU_counts':cpu['counts'],'coefficient_replay_drift':r['coefficient_replay_drift_report_only'],
    'GPU_peak_allocated_bytes':r['peak_allocated_GPU_bytes'],'GPU_peak_reserved_bytes':r['peak_reserved_GPU_bytes'],
    'additional_actual_tensor_CPU_bytes':r['retained_tensor_CPU_bytes'],'points':{},
    'audit_scope':'Independently rehash downloaded source/protocol/results/NPZ/terminal receipts, recompute signed stats, per-head/group sums and per-token identities, and reconnect to preceding real FLA boundary. Remote multi-GB operands and additional actual native-adjoint private tensors are not downloaded/recomputed here; their recorded hashes are checked against source receipts. No GPU/model/metric call.'}
for step in ('1','10','20','B2'):
    row=cpu['points'][step];assert set(row['contractions'])==set(term_names+aux_names)
    deleted=keep if step=='B2' else set(boundary['points'][step]['input_receipt']['deleted_positions'])
    assert deleted<=keep
    groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    arrays={name:z[step+'_'+name] for name in term_names+aux_names}
    max_stat_error=0.
    for name,x in arrays.items():
        assert x.shape==(1,T,H) and x.dtype==np.float64 and np.isfinite(x).all()
        recorded=row['contractions'][name];actual=stats(x)
        assert len(recorded['heads'])==H
        for k in actual:max_stat_error=max(max_stat_error,close(actual[k],recorded['total'][k]))
        for h in range(H):
            for k,value in stats(x[:,:,h]).items():max_stat_error=max(max_stat_error,close(value,recorded['heads'][h][k]))
        for group,positions in groups.items():
            assert recorded['groups'][group]['count']==len(positions)
            for k,value in stats(x[:,positions]).items():max_stat_error=max(max_stat_error,close(value,recorded['groups'][group][k]))
        for k in actual:
            max_stat_error=max(max_stat_error,close(sum(v[k] for v in recorded['heads']),actual[k]))
            max_stat_error=max(max_stat_error,close(sum(v[k] for v in recorded['groups'].values()),actual[k]))
    reconstruction=sum(arrays[k] for k in term_names)
    token_closure=close(reconstruction,arrays['saved_error'])
    close(reconstruction-arrays['saved_error'],arrays['closure_residual'])
    close(arrays['MA1_primitive_prediction']-arrays['B2input_minus_A_actual_output'],arrays[remainder_name])
    assert not np.count_nonzero(arrays['v_reference_difference'])
    if step=='B2':
        assert not np.count_nonzero(arrays[transfer_name])
        assert all(not np.count_nonzero(arrays[k]) for k in reference_names)
    else:assert np.array_equal(arrays[transfer_name],z['1_'+transfer_name])
    previous=source['conditional_ledgers'][step]['replayed_16term_ledger']['terms']['GDN_FLA_including_raw_g_exp']
    close(arrays['saved_error'].sum(),previous);close(row['saved_error']['net'],previous)
    terms={name:stats(arrays[name]) for name in term_names}
    paired=float((arrays['saved_error']-arrays[transfer_name]).sum())
    out['points'][step]={'saved_FLA_error':stats(arrays['saved_error']),'matched_B2input_A_saved_error':paired,
        'signed_terms':terms,'reference_sum':stats(sum(arrays[k] for k in reference_names)),
        'reference_terms_by_absolute_net':sorted([(k,terms[k]['net']) for k in reference_names],key=lambda kv:-abs(kv[1])),
        'positive_terms_ranked':sorted([(k,v['net']) for k,v in terms.items() if v['net']>0],key=lambda kv:-kv[1]),
        'negative_terms_ranked':sorted([(k,v['net']) for k,v in terms.items() if v['net']<0],key=lambda kv:kv[1]),
        'saved_to_replayed_transfer':stats(sum(arrays['saved_to_replayed_GPU_'+k] for k in branches)),
        'replayed_to_CPU_transfer':stats(sum(arrays['replayed_GPU_to_CPU01_'+k] for k in branches)),
        'conditional_exp_secant':terms[exp_name],'matched_pair_remainder':terms[remainder_name],
        'common_B1_B2_transfer':terms[transfer_name],'per_token_head_closure_max_absolute':token_closure,
        'head_group_statistics_max_difference':max_stat_error,
        'group_net_by_reference_term':{k:{g:row['contractions'][k]['groups'][g]['net'] for g in groups} for k in reference_names}}
mid=out['points']['10'];early=out['points']['1']
largest=mid['reference_terms_by_absolute_net'][0]
out['measured_priority']={'mid_largest_absolute_net_reference_term':{'name':largest[0],'value':largest[1]},
    'mid_largest_positive_term':mid['positive_terms_ranked'][0] if mid['positive_terms_ranked'] else None,
    'mid_largest_negative_term':mid['negative_terms_ranked'][0] if mid['negative_terms_ranked'] else None,
    'early_same_reference_term':early['signed_terms'][largest[0]],
    'scope':'Largest reference allocation is selected from actual signed term values. Sign and cancellation, early versus middle, endpoint controls, numerical transfers and remaining paired residual must be assessed jointly. Arithmetic closure proves ledger completeness only, not a better method or an independently causal source.'}
out['interpretation_limits']=['MA1 is coefficient algebra from real retained A/1 states with the same actual endpoint1 adjoints; it is not a new model counterfactual.',
    'Query/key/beta/decay values quantify the current finite-rule reference change, not independent original-token causal effects or proven unique fixes.',
    'The alpha reference term and exp-secant conditional term are separate; g total cannot all be called a forgetting-state error.',
    'Matched-pair residual retains native precision/chunk/adjoint-consistency effects; it is not silently assigned to a reference branch.',
    'No new RISE, MAS, needle, candidate effectiveness or production efficiency is established.']
out['audit_seconds']=time.perf_counter()-started
target=A/'dt_NI_fla_reference_mismatch_summary_20260908.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'output':str(target),'sha256':sha(target),'mid_error':mid['saved_FLA_error']['net'],
    'mid_reference_terms':mid['reference_terms_by_absolute_net'],'mid_exp':mid['conditional_exp_secant']['net'],
    'mid_remaining':mid['matched_pair_remainder']['net'],'audit_seconds':out['audit_seconds']}))
