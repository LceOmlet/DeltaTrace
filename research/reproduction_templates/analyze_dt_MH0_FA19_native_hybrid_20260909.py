"""Independent NumPy audit of the current MH0 eleven native FA contrasts."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_native_hybrid_20260909_v1'
started=time.perf_counter();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(x,y,label='',tol=1e-7):
    err=float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)),initial=0));assert err<tol,(label,err)
    return err
def stats(x):
    return {'net':float(x.sum()),'positive_sum':float(x.clip(min=0).sum()),'negative_sum':float(x.clip(max=0).sum()),'absolute_sum':float(np.abs(x).sum()),
        'positive_token_count':int((x>0).sum()),'negative_token_count':int((x<0).sum())}
p,r=read(D/'protocol.json'),read(D/'results.json');assert p==r['protocol']==read(A/'dt_MH0_FA19_native_hybrid_protocol_20260909.json')
assert sha(D/'protocol.json')=='b38b819f6564a800f9ba3879ef915a2859b4ab3c6a525f7f659916055b685de0'
assert r['status']=='MH0_eleven_native_FA_hybrid_contrasts_complete'
receipt=read(D/'terminal_receipt.json');assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():
    assert sha(D/name)==(item if isinstance(item,str) else item['sha256']),name
    if isinstance(item,dict) and 'bytes' in item:assert (D/name).stat().st_size==item['bytes']
for name,want in p['files_sha256'].items():assert sha(D/name)==want
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    for name in archive.namelist():assert archive.read(name)==(D/name).read_bytes()
assert r['native_FA_calls_entered']==r['native_FA_calls_returned']==11
assert [c['label'] for c in r['calls']]==p['native_call_schedule']
assert all(c['status']=='returned' and c['kwargs']==p['native_FA_kwargs'] for c in r['calls'])
assert all(r[k]==0 for k in ['model_calls','DT_calls','scorer_calls','FT_calls','backward_calls','generation_calls'])
assert r['private_artifact_sha256_before']==r['private_artifact_sha256_after']==p['private_artifact_sha256']
assert r['FA_interface_sha256_before']==r['FA_interface_sha256_after']==p['installed_FA_interface_sha256']
S=(A/'snapshot'/p['source_results_path'].lstrip('/')).parent;assert sha(S/'results.json')==p['source_results_sha256'];source=read(S/'results.json')
B=A/'snapshot'/p['source_boundary_results_path'].lstrip('/');assert sha(B)==p['source_boundary_results_sha256'];boundary=read(B)
assert source['input']==boundary['cases']['morehopqa_0']['input'];T,P=source['input']['total_length'],source['input']['prompt_length'];keep=set(source['input']['keep'])
assert T==853 and r['FA_layout']['padded_length']==T and r['FA_layout']['valid_lengths']==[T]
meta=r['FA_layout'];assert meta['query_heads']==meta['kv_heads']*meta['groups'] and meta['mapping']=='query_head // groups = compact_kv_head'
assert p['fixed_steps']==['3','10','20'] and set(r['replays'])=={'0','3','10','20'}
for step,entry in r['prepared_GPU_operand_layouts'].items():
    for x in ([entry] if step=='paired_EOS_value' else entry.values()):assert x['dtype']=='torch.bfloat16' and x['is_cuda'] and x['contiguous']
assert sha(D/'signed_token_contrasts.npz')==r['signed_token_contrasts']['sha256']
z=np.load(D/'signed_token_contrasts.npz',allow_pickle=False)
terms=['routing_at_baseline_values_prediction_error','content_at_clean_routing_prediction_error','routing_content_interaction_contrast']
out={'status':'MH0_FA19_native_hybrid_independent_CPU_audit_passed','next':'Reuse the existing compiled QK/softmax diagnostic to partition routing-at-V0 while retaining the larger measured PV-reference interaction; no candidate or new kernel.',
    'analyzer_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),
    'signed_token_contrasts_sha256':sha(D/'signed_token_contrasts.npz'),'source_results_sha256':p['source_results_sha256'],
    'actual_budget':{'native_FA_entered':11,'native_FA_returned':11,'model':0,'DT':0,'scorer':0,'FT':0,'backward':0,'generation':0,
        'job_seconds':r['job_seconds'],'audit_model_calls':0,'failed_entered_minus_returned':0,'scope':'This operator diagnostic only; no quality metric or production cost claim.'},
    'replays':r['replays'],'saved_B1_vs_B2_endpoint_drift':r['saved_B1_vs_B2_endpoint_drift'],'points':{},'FA_layout':meta,
    'precision_contract':r['precision_contract'],'group_contract':r['group_contract'],
    'proof_scope':'Public per-token identities, group/net signs, stored-core connection and source/native/layout receipts audited. Remote private operands and coefficients were not dot-product recomputed locally; native job recomputed the existing core before FA calls.'}
keys=set()
for step,row in r['points'].items():
    assert row['input_receipt']==p['frozen_input_receipts'][step]==boundary['cases']['morehopqa_0']['points'][step]['input_receipt']
    fields={name:z[step+'_'+name] for name in row['fields']};keys.update(step+'_'+name for name in fields)
    deleted=set(row['input_receipt']['deleted_positions']);groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
    for name,x in fields.items():
        assert x.shape==(1,T) and x.dtype==np.float64 and np.isfinite(x).all()
        for key,value in stats(x).items():close(value,row['fields'][name][key])
        for group,indices in groups.items():
            assert row['groups'][group]['count']==len(indices)
            for key,value in stats(x[:,indices]).items():close(value,row['groups'][group]['fields'][name][key])
    core=fields['core_error_at_stored_seed'];error=source['conditional_ledgers'][step]['replayed_9term_ledger']['terms']['finite_FA_core_including_seed_cast']
    close(core.sum(),error);close(row['original_FA_core_error'],error);close(row['CPU_recomputed_original_FA_core_error'],error)
    maximum=close(sum(fields[k] for k in terms),core,'three term token closure')
    close(fields['qk_prediction']-fields['R0'],fields[terms[0]])
    close(fields['v_prediction']-fields['VCVA'],fields[terms[1]])
    close(fields['R0']-fields['RA'],fields[terms[2]])
    close(fields['RA']+fields['VCVA'],fields['captured_actual_effect'])
    close(core,fields['qk_prediction']+fields['v_prediction']-fields['captured_actual_effect'])
    close(core-fields['core_error_at_BF16_seed'],fields['BF16_minus_stored_seed_actual_effect'])
    out['points'][step]={'core_error':error,'terms':{k:row['fields'][k] for k in terms},'seed_cast':row['fields']['BF16_minus_stored_seed_actual_effect'],
        'groups':row['groups'],'max_token_three_term_closure':maximum,
        'largest_absolute_three_term':max(terms,key=lambda k:abs(row['fields'][k]['net']))}
assert set(z.files)==keys
out['interpretation']='Current early/mid PV-reference routing interaction is larger than routing-at-V0 error. The value-allocation-at-clean-routing term is small; this does not mean the reference-value interaction is negligible. Negative/positive token and group cancellation is retained. Replay or BF16 seed conversion magnitude cannot explain the large terms. Operator hybrids are not separately executed model counterfactuals and imply no repair or MAS benefit.'
out['CPU_audit_seconds']=time.perf_counter()-started
path=A/'dt_MH0_FA19_native_hybrid_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'summary':str(path),'sha256':sha(path),'points':{s:{'core':q['core_error'],**{k:v['net'] for k,v in q['terms'].items()},'seed_cast':q['seed_cast']['net']} for s,q in out['points'].items()},'seconds':out['CPU_audit_seconds']}))
