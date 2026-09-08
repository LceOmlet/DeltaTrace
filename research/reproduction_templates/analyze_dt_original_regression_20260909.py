"""CPU-only independent audit of pristine FT/current DT and fixed eight-case shards.

Reads retained vectors, original scoring receipts and source-pinned formulas.
Does not import torch, tokenize again, run a model or substitute historical scores.
"""
import argparse,hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
digest=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(Path(path).read_bytes())
local=lambda path:A/'snapshot'/str(path).lstrip('/')
METHODS=('DT_control','DT_candidate','FT0','FT3')

def close(actual,expected,tol=1e-9):
    aa=np.asarray(actual);bb=np.asarray(expected)
    assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
    assert np.max(np.abs(aa-bb),initial=0)<=tol,(actual,expected,tol)

def stats(x):
    x=np.asarray(x,dtype=np.float64)
    assert np.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x[x>0].sum()),'negative':float(x[x<0].sum()),'absolute':float(np.abs(x).sum())}

def auc(x):return float((x.sum()-x[0]/2-x[-1]/2)/max(1,len(x)-1))

def original_metric(scores,density):
    """Exact original loops, including degenerate corrected-curve fallback."""
    scores=np.asarray(scores,dtype=np.float64);density=np.asarray(density,dtype=np.float64)
    minimum=1.0;response=scores.copy()
    with np.errstate(divide='ignore',invalid='ignore'):
        for i in range(len(scores)):
            normalized=np.clip((response[i]-scores[-1])/abs(scores[0]-scores[-1]),0.0,1.0)
            minimum=min(minimum,normalized);response[i]=minimum
        penalty=np.abs(response-density);corrected=np.clip(response+penalty,0.0,1.0)
        corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
    fallback=bool(np.isnan(corrected).any())
    if fallback:corrected=np.linspace(1.0,0.0,len(scores))
    return response,penalty,corrected,[auc(response),auc(corrected),auc(response+penalty)],fallback

def gamma(n):return n*2**-24/(1-n*2**-24)


def mas_decomposition(scores,density,response,penalty,corrected,values,dt_errors=None):
    """An exact metric ledger, without editing or re-scoring any attribution."""
    negative=np.maximum(-density,0.0)
    remainder=np.abs(response-np.maximum(density,0.0))
    # R is nonnegative after the author's clipping/envelope.
    close(penalty,negative+remainder)
    ap=auc(penalty);neg=auc(negative);rest=auc(remainder)
    pre=response+penalty;clipped=np.clip(pre,0.0,1.0)
    clip_change=auc(clipped)-auc(pre);normalize_change=auc(corrected)-auc(clipped)
    post=clip_change+normalize_change
    close(values[1],values[0]+neg+rest+post)
    drop=abs(float(scores[0]-scores[-1]));assert drop>0
    raw=(scores-scores[-1])/drop
    negmask=density<0
    output={'RISE':values[0],'AP_AUC':ap,'negative_density_AP_extra_AUC':neg,
        'AP_with_nonnegative_density_AUC':rest,'original_clip_change_AUC':clip_change,
        'original_normalization_or_fallback_change_AUC':normalize_change,
        'original_postprocessing_change_AUC':post,'MAS':values[1],
        'closure':values[1]-(values[0]+neg+rest+post),
        'negative_density_points':np.flatnonzero(negmask).tolist(),
        'negative_density_min':float(density.min()),
        'raw_response_negative_at_negative_density_points':np.flatnonzero(negmask&(raw<0)).tolist(),
        'negative_density_below_actual_raw_response_points':np.flatnonzero(negmask&(density<raw)).tolist(),
        'negative_density_normalized_overprediction_signed_AUC':auc(np.where(negmask,raw-density,0.0)),
        'point_ledger':[{'step':i,'raw_score':float(scores[i]),'raw_normalized_response':float(raw[i]),
            'author_response':float(response[i]),'density':float(density[i]),'AP':float(penalty[i]),
            'negative_density_AP_extra':float(negative[i]),'remaining_AP':float(remainder[i])} for i in range(21)],
        'scope':'Identity AP=abs(R-rho)=max(-rho,0)+abs(R-max(rho,0)), since R>=0. This is a CPU decomposition, not a clipped-attribution candidate. Negative density means the remaining signed attribution is negative; it does not by itself prove negative token contributions wrong. Raw scores are retained to detect genuine below-baseline responses hidden by the original clipping/envelope. Methods use their own deletion sets.'}
    if dt_errors is not None:
        errors=np.asarray(dt_errors,dtype=np.float64)
        output['DT_raw_logprob_overprediction_at_negative_density']={
            'steps':np.flatnonzero(negmask).tolist(),'errors':errors[negmask].tolist(),
            'positive_error_steps':np.flatnonzero(negmask&(errors>0)).tolist(),
            'signed_AUC':auc(np.where(negmask,errors,0.0)),
            'formula':'sum frozen signed DT scores over actual deleted positions - (raw clean logprob - raw deleted logprob). Positive is conditional deletion-effect overprediction, not a negative causal sign.'}
    return output

def needle(w,keep,gold,reported):
    gold=set(gold)&set(keep)
    if not gold:assert reported is None;return None
    # Clamping is only the unchanged author's needle metric, not a vector edit.
    ww=np.maximum(w[keep].astype(np.float32),0);k=max(1,min(len(keep),math.ceil(len(keep)*.1)))
    threshold=np.sort(ww)[-k]
    above={keep[i] for i in np.flatnonzero(ww>threshold)}
    ties={keep[i] for i in np.flatnonzero(ww==threshold)};slots=k-len(above)
    lo=(len(above&gold)+max(0,slots-len(ties-gold)))/len(gold)
    hi=(len(above&gold)+min(slots,len(ties&gold)))/len(gold)
    assert lo-1e-12<=reported<=hi+1e-12
    hits=reported*len(gold);assert abs(hits-round(hits))<1e-9
    return {'reported':reported,'hits':int(round(hits)),'eligible_gold_denominator':len(gold),
        'top_k':k,'tie_aware_interval':[lo,hi],'threshold_tie_count':len(ties)}

def masks(w,info,ids,eos,audit):
    keep=info['keep'];order=audit['sorted_keep'];assert sorted(order)==sorted(keep)
    assert np.all(np.diff(w[order].astype(np.float64))<=0)
    groups=audit['groups'];receipts=audit['input_receipts'];assert len(groups)==20 and len(receipts)==21
    n,rem=divmod(len(keep),20);cursor=0;deleted=set()
    for i,receipt in enumerate(receipts):
        if i:
            group=order[cursor:cursor+n+(i<=rem)];cursor+=len(group)
            assert group==groups[i-1];deleted.update(group)
        x=ids.copy();x[sorted(deleted)]=eos
        assert receipt=={'input_sha256':digest(x[None].tobytes()),'deleted_positions':sorted(deleted)}
    assert deleted==set(keep)

def source_audit(D,p,r,receipt):
    for key in ('pid_alive','proc_exists'):
        if key in receipt:assert receipt[key] is False
    for name,row in receipt['files'].items():
        want=row if isinstance(row,str) else row['sha256'];assert sha(D/name)==want,name
        if isinstance(row,dict) and 'bytes' in row:assert (D/name).stat().st_size==row['bytes']
    for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
    with zipfile.ZipFile(D/'review_bundle.zip') as bundle:
        for name in bundle.namelist():assert bundle.read(name)==(D/name).read_bytes(),name
    assert r['sources_before']==r['sources_after']
    assert r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
    source_root=local(p['official_root'])
    actual={str(f.relative_to(source_root)).replace('\\','/') for f in (source_root/'flashtrace').rglob('*.py')}
    assert actual==set(p['official_package_blob_sha1']) and len(actual)==16
    for name,want in p['official_source_blob_sha1'].items():
        raw=(source_root/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        assert r['sources_before']['official/'+name]==digest(raw)
    for name,want in p['files_sha256'].items():assert r['sources_before'][name]==want
    for name,want in p['runtime_source_sha256'].items():assert r['sources_before']['native/'+name]==want
    for row in p['protected_sources']:assert r['sources_before'][row['path']]==row['sha256']
    for name,want in p['span_source_sha256'].items():
        raw=(local(p['author_data_root'])/name).read_bytes().replace(b'\r\n',b'\n')
        assert digest(raw)==want==r['sources_before']['author_spans/'+name]
    assert p['FT_commit']=='e81b3be50a48dcfc652fbf1b530069b552736e66'
    if 'segment_index' in p:
        assert sha(D/'regression_manifest.json')==p['regression_manifest_sha256']
        assert read(D/'regression_manifest.json')==read(A/'dt_fixed_eight_case_regression_manifest_20260909.json')
        assert r['sources_before']['regression_manifest.json']==p['regression_manifest_sha256']
    return {'official_package_file_count':16,'official_commit':p['FT_commit'],
        'official_metric_blob':p['official_source_blob_sha1']['flashtrace/improved.py'],
        'source_before_after_equal':True,'receipt_files_verified':len(receipt['files']),
        'scope':'Independent local original FT16-package Git-blob and artifact checks; runtime native/weight before-after receipts verified against frozen expected identities. No model or native operator reexecution.'}

def input_audit(D,p,r):
    inputs={};out={}
    for dataset,index in p['case_indices']:
        key=f'{dataset}_{index}';case=r['cases'][key];info=case['input'];frozen=r['input_freeze_before_model_load'][key]
        spec=p['fixed_records'][key];raw=local(p['cache_paths'][dataset]).read_bytes()
        assert digest(raw)==p['cache_hashes'][dataset]
        line=raw.decode().splitlines()[index];rec=json.loads(line)
        assert digest(line.encode())==spec['source_record_sha256']==case['mapping']['source_record_sha256']
        assert digest(rec['prompt'].encode())==spec['prompt_text_sha256']
        assert digest(rec['target'].encode())==spec['fixed_target_text_sha256']
        assert frozen['input']==info and frozen['mapping']==case['mapping']
        assert case['mapping']['original_cached_spans_reproduced'] is True
        assert case['mapping']['original_sink_span']==rec['sink_span']
        assert case['mapping']['gold_function']=='unchanged author ruler_gold_prompt_token_indices'
        assert case['mapping']['gold']==case['gold']
        if spec['expected_input'] is not None:assert info==spec['expected_input']
        if spec['expected_gold'] is not None:assert case['gold']==spec['expected_gold']
        if 'expected_lengths' in spec:assert {k:info[k] for k in spec['expected_lengths']}==spec['expected_lengths']
        if not rec['metadata'].get('needle_spans'):assert case['gold']==[]
        ids=np.asarray(frozen['input_ids'],dtype=np.int64);target=np.asarray(frozen['target_ids'],dtype=np.int64)
        assert ids.shape==(info['total_length'],) and target.shape==(info['target_length'],)
        assert np.array_equal(ids[info['prompt_length']:],target)
        assert digest(ids.tobytes())==info['input_sha256']
        base=ids.copy();base[info['keep']]=target[-1]
        assert digest(base.tobytes())==frozen['baseline_sha256']==case['baseline_sha256']
        assert np.flatnonzero(base!=ids).tolist()==info['keep']
        inputs[key]=(ids,int(target[-1]))
        out[key]={'source_record_sha256':spec['source_record_sha256'],'input_sha256':info['input_sha256'],
            'prompt_length':info['prompt_length'],'target_length':info['target_length'],'total_length':info['total_length'],
            'eligible_count':len(info['keep']),'gold':case['gold'],'sink_span':case['mapping']['new_sink_span'],
            'thinking_span':case['mapping']['new_thinking_span'],
            'scope':'Actual saved IDs/target suffix/EOS baseline and author cache hashes independently verified. Original author span/gold source and runtime remap receipts verified; tokenizer is not rerun locally.'}
    return inputs,out

def dt_audit(run,p,r,vectors,inputs):
    assert run['status']=='complete' and run['root_forwards']==1
    extra=int(run['method'] in ('DT','DT_candidate'))
    counts=run['counts'];d=run['details']
    assert {k:counts[k] for k in ('native_root','native_decoder_replays','finite_decoder_calls','public_FA_auxiliary_calls','finite_FA_calls','finite_FLA_calls')}=={
        'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24}
    assert counts['finite_FLA_backend_calls']==24+extra
    assert run['finite_callback_counts']=={'FA_entered':8,'FA_returned':8,'FLA_backend_entered':24+extra,'FLA_backend_returned':24+extra}
    assert run['native_stage_accounting']=={'returned_FLA_backend_calls_times_two':2*(24+extra),'stages_inside_nonreturned_backend':0}
    assert d['norm_gate_rules']=={'0':'symmetric'} and d['finite_fla_by_layer']==([0] if extra else []) and d['select_output_rows'] is True
    assert len(run['candidate_layer0_receipts'])==extra
    kinds=[x['kind'] for x in d['calls']]
    for prefix in ('native_replay_','finite_decoder_'):
        assert [x for x in kinds if x.startswith(prefix)]==[prefix+str(i) for i in reversed(range(32))]
    assert [x for x in kinds if x.startswith('public_FA_LSE_')]==['public_FA_LSE_'+str(i) for i in reversed(p['expected_FA_layers'])]
    assert list(d['layers'])==[str(i) for i in reversed(range(32))]
    for layer in d['layers'].values():
        assert layer['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','silu','up','down','mlp','decoder')}
        assert layer['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if layer['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
    key=run['case'];info=r['cases'][key]['input'];vk=run['vector_key'];full=vectors[vk+'_full'];w=vectors[vk+'_evaluated']
    assert full.dtype==np.float64 and w.dtype==np.float32
    assert full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],) and np.array_equal(full[:len(w)].astype(np.float32),w)
    close(float(full.sum()),d['signed_sum'],1e-7)
    for field in ('net','positive','negative'):close(stats(full)[field],run['signed_summary'][field],1e-7)
    needle(w,info['keep'],r['cases'][key]['gold'],run['signed_summary']['needle'])
    ids,eos=inputs[key];masks(w,info,ids,eos,run['deletion_audit'])
    mem=run['memory_cost'];assert mem['peak_allocated_full_model_resident']==d['peak_allocated']
    assert mem['peak_reserved_full_model_resident']==d['peak_reserved']
    assert mem['peak_allocated_minus_before_pair']==d['peak_allocated']-run['GPU_allocated_before_pair']
    assert mem['peak_allocated_minus_before_attribute']==d['peak_allocated']-run['GPU_allocated_before_attribute']
    return {'case':key,'method':run['method'],'sample_batch':1,'endpoint_batch':2,'outer_seconds':run['outer_attribute_seconds'],
        'runner_seconds':d['complete_attribution_seconds_with_diagnostics'],'peak_allocated':d['peak_allocated'],
        'peak_reserved':d['peak_reserved'],'memory_cost':mem,'root_effect':d['root_effect'],'seed_effect':d['seed_effect'],
        'signed_sum':d['signed_sum'],'relative_residual':d['relative_residual'],'finite_FLA_calls':24+extra,
        'scope':'Actual single-call diagnostic-inclusive attribution cost; not minibatch or warmed speed comparison.'}

def ft_audit(key,case,inputs,vectors,scored):
    ft=case['FT'];info=case['input'];ids,eos=inputs[key];counts=ft['counts']
    assert ft['status']=='complete' and ft['root_entered']==ft['root_returned']==1
    assert ft['input_receipts']==[{'input_sha256':info['input_sha256'],'shape':[1,info['total_length']]}]
    for name,want in {'official.calculate_ifr_multi_hop_both':1,'official._capture_model_state':1,
        'native_FLA.chunk_gated_delta_rule':72,'native_model.eager_attention_forward':8,'official.build_layer_inputs':1,
        'official._linear_layer_input':24,'official._full_layer_input':8}.items():assert counts[name]==want
    assert counts['official.compute_ifr_sentence_aggregate']<=4
    events=ft['events'];assert len(events)==72 and all(x['kind']=='native_FLA_call' for x in events)
    probe=[x for x in events if x['value_shape'][-1]==info['total_length']]
    assert len(probe)==24 and all(x['value_shape']==[1,info['total_length'],32,info['total_length']] for x in probe)
    assert all(x['value_shape']==[1,info['total_length'],32,128] for x in events if x not in probe)
    assert len(ft['attention_shapes'])==8
    meta=ft['official_span_metadata'];assert meta['n_hops']==3
    assert meta['sink_span_generation']==case['mapping']['new_sink_span']
    assert meta['thinking_span_generation']==case['mapping']['new_thinking_span']
    assert meta['all_gen_span_generation']==[0,info['target_length']-2]
    assert 'both = stop_words + in_all_gen' in meta['note']
    saved=[]
    for hop in range(4):
        vk=f'{key}_FT{hop}';full=vectors[vk+'_full'];w=vectors[vk+'_evaluated']
        assert full.dtype==w.dtype==np.float32 and full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],)
        assert np.array_equal(full[:len(w)],w) and np.isfinite(full).all()
        if hop in scored:
            audit=ft['deletion_audit'] if len(scored)==1 else ft['deletion_audits'][f'FT{hop}']
            masks(w,info,ids,eos,audit)
        saved.append({'hop':hop,'scored':hop in scored,'full_vector_sha256':digest(full.tobytes()),'full_stats':stats(full)})
    return {'case':key,'sample_batch':1,'complete_Both_seconds_with_passive_counts':ft['seconds_with_passive_counts'],
        'peak_allocated':ft['peak_allocated'],'peak_reserved':ft['peak_reserved'],'counts':counts,
        'identity_probe_count':24,'saved_hops':saved,'scope':'Entire unchanged original Both trace(hops3), including native identity probes and passive profile overhead; not isolated FT0 cost.'}

def curve_audit(key,method,curve,case,audit,vectors,inputs):
    info=case['input'];ids,eos=inputs[key];keep=info['keep'];K=len(keep)
    full=vectors[key+'_'+method+'_full'];w=vectors[key+'_'+method+'_evaluated']
    assert full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],)
    assert np.array_equal(full[:len(w)].astype(np.float32),w) and np.isfinite(full).all()
    assert curve['status']=='complete' and curve['returned_forwards']==21
    assert curve['sorted_keep']==audit['sorted_keep'] and curve['input_receipts']==audit['input_receipts']
    masks(w,info,ids,eos,audit)
    scores=np.asarray(curve['scores'],dtype=np.float64);density=np.asarray(curve['density'],dtype=np.float64)
    assert scores.shape==density.shape==(21,) and np.isfinite(scores).all() and np.isfinite(density).all()
    total=curve['attr_sum'];exact_total=float(w[keep].astype(np.float64).sum())
    total_bound=gamma(K)*float(np.abs(w[keep].astype(np.float64)).sum())
    assert abs(total-exact_total)<=total_bound+1e-12
    groups=audit['groups']
    predicted_density=np.linspace(1,0,21) if total<=0 else np.r_[1,1-np.cumsum([w[g].astype(np.float64).sum()/total for g in groups])]
    density_bound=1e-12 if total<=0 else sum(gamma(len(g))*float(np.abs(w[g].astype(np.float64)).sum())/abs(total) for g in groups)+1e-12
    maxdelta=float(np.max(np.abs(density-predicted_density)));assert maxdelta<=density_bound
    response,penalty,corrected,values,fallback=original_metric(scores,density)
    for name,x in [('normalized_model_response',response),('alignment_penalty',penalty),('corrected_scores',corrected),('return_metrics',values)]:close(curve[name],x)
    answer={'RISE':values[0],'MAS':values[1],'alignment_augmented_AUC':values[2],
        'needle':needle(w,keep,case['gold'],curve['needle']),'scores':scores.tolist(),'density':density.tolist(),
        'normalized_model_response':response.tolist(),'alignment_penalty':penalty.tolist(),'corrected_scores':corrected.tolist(),
        'original_constant_corrected_curve_fallback':fallback,'attr_sum_reported_FP32':total,'attr_sum_exact':exact_total,
        'density_recompute_max_delta':maxdelta,'density_analytical_rounding_bound':density_bound,
        'eligible_signed':stats(w[keep]),'full_signed':stats(full),'response_signed':stats(full[info['prompt_length']:]),
        'sorted_keep':curve['sorted_keep'],'input_receipts':curve['input_receipts']}
    if method.startswith('DT'):
        answer['own_masks_prediction_minus_actual']=[float(w[x['deleted_positions']].astype(np.float64).sum()-(scores[0]-scores[i])) for i,x in enumerate(curve['input_receipts'])]
    else:answer['units_note']='FT representation scores are not logprob finite multipliers; do not subtract their raw sum from a logprob drop as an equivalent-unit attribution error.'
    answer['MAS_decomposition']=mas_decomposition(scores,density,response,penalty,corrected,values,answer.get('own_masks_prediction_minus_actual'))
    return answer

def audit_directory(D):
    started=time.perf_counter();D=Path(D);p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
    assert r['protocol']==p
    segment=p.get('segment_index');shard=segment is not None
    status='fixed_eight_case_regression_shard_2cases4DT2Both168score_complete' if shard else 'MH0_current_DT_pristine_FT0_1DT1Both42score_complete'
    assert r['status']==status,r['status']
    frozen=A/(f'dt_fixed_eight_case_regression_protocol_20260909_s{segment}.json' if shard else 'dt_MH0_pristine_FT0_comparison_protocol_20260909.json')
    assert p==read(frozen)
    sources=source_audit(D,p,r,receipt);inputs,provenance=input_audit(D,p,r)
    ndt=4 if shard else 1;nft=2 if shard else 1;nscore=168 if shard else 42;nextra=2 if shard else 1
    assert r['model_loads']==r['native_eager_diagnostics']==1 and r['generation_calls']==0
    assert r['DT_entered']==r['DT_returned']==ndt and r['FT_entered']==r['FT_calls']==nft
    assert r['scorer_entered']==r['scorer_returned']==nscore
    assert len(r['runs'])==ndt and [[x['case'],x['method']] for x in r['runs']]==p['call_schedule']
    expected_methods=('DT_control','DT_candidate') if shard else ('DT',)
    for method in expected_methods:
        n=sum(x['method']==method for x in r['runs']);assert r['finite_counts'][method]=={'entered':8*n,'returned':8*n}
    nfla=24*ndt+nextra
    assert r['finite_counts']['FLA_backend']=={'entered':nfla,'returned':nfla,'native_adjoint_stages_from_returned_calls':2*nfla,'native_stages_inside_nonreturned_calls':0}
    wrapped=r['finite_counts']['layer0_average_wrapper'];assert wrapped['entered']==wrapped['returned']==len(wrapped['calls'])==nextra
    for row in wrapped['calls']:
        assert row['status']=='returned' and row['endpoint_permutation']==[1,0] and row['do_dtype']=='torch.bfloat16'
        assert [x['orientation'] for x in row['backend_calls']]==['original','swapped'] and all(x['status']=='returned' for x in row['backend_calls'])
    assert sha(D/'vectors.npz')==r['vectors_sha256'];vectors=np.load(D/'vectors.npz',allow_pickle=False)
    out={'status':'independent_original_FT_DT_audit_passed','segment_index':segment,'analyzer_sha256':sha(__file__),
        'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),
        'sources':sources,'input_provenance':provenance,'wall_seconds':r['seconds'],'cases':{},
        'counts':{'model_loads':1,'eager_initializations':1,'DT':ndt,'DT_native_decoder_replays':32*ndt,'finite_FA':8*ndt,'public_LSE':8*ndt,
            'finite_FLA':nfla,'native_adjoint_stages':2*nfla,'FT_complete_Both':nft,'FT_native_FLA':72*nft,'FT_identity_probes':24*nft,'original_scores':nscore,'generation':0},
        'DT_costs':[dt_audit(x,p,r,vectors,inputs) for x in r['runs']],'FT_costs':[]}
    keys=p['quality_cases'] if shard else ['morehopqa_0'];scored=[0,3] if shard else [0]
    expected_vector_keys={key+'_'+m+'_'+part for key in keys for m in [*expected_methods,'FT0','FT1','FT2','FT3'] for part in ('full','evaluated')}
    assert set(vectors.files)==expected_vector_keys
    for key in keys:
        case=r['cases'][key];out['FT_costs'].append(ft_audit(key,case,inputs,vectors,scored))
        methods=METHODS if shard else ('DT','FT0');assert set(case['curves'])==set(methods)
        result={'input_sha256':case['input']['input_sha256'],'methods':{},'candidate_minus_comparators':{}}
        for method in methods:
            if method.startswith('FT'):a=case['FT']['deletion_audits'][method] if shard else case['FT']['deletion_audit']
            else:a=next(x['deletion_audit'] for x in r['runs'] if x['case']==key and x['method']==method)
            result['methods'][method]=curve_audit(key,method,case['curves'][method],case,a,vectors,inputs)
        candidate='DT_candidate' if shard else 'DT'
        for ref in methods:
            if ref==candidate:continue
            x=result['methods'][candidate];y=result['methods'][ref]
            result['candidate_minus_comparators'][ref]={k:x[k]-y[k] for k in ('RISE','MAS','alignment_augmented_AUC')}
            result['candidate_minus_comparators'][ref]['needle']=x['needle']['reported']-y['needle']['reported'] if x['needle'] is not None else None
            parts=('RISE','negative_density_AP_extra_AUC','AP_with_nonnegative_density_AUC','original_postprocessing_change_AUC')
            result['candidate_minus_comparators'][ref]['MAS_decomposition']={k:x['MAS_decomposition'][k]-y['MAS_decomposition'][k] for k in parts}
            close(sum(result['candidate_minus_comparators'][ref]['MAS_decomposition'].values()),x['MAS']-y['MAS'])
            for step in (0,20):assert x['input_receipts'][step]==y['input_receipts'][step]
        result['actual_endpoint_scores']={m:[case['curves'][m]['scores'][i] for i in (0,20)] for m in methods}
        result['endpoint_score_drift_against_candidate']={m:[result['actual_endpoint_scores'][m][i]-result['actual_endpoint_scores'][candidate][i] for i in (0,1)] for m in methods}
        if shard:
            assert result['actual_endpoint_scores']==case['actual_endpoint_scores']
            for ref,row in result['candidate_minus_comparators'].items():
                close(case['candidate_minus_comparators'][ref]['return_metrics'],[row[k] for k in ('RISE','MAS','alignment_augmented_AUC')])
                if row['needle'] is not None:close(row['needle'],case['candidate_minus_comparators'][ref]['needle'])
        else:close(case['DT_minus_FT0_return_metrics'],[result['candidate_minus_comparators']['FT0'][k] for k in ('RISE','MAS','alignment_augmented_AUC')])
        out['cases'][key]=result
    out['target_scope']=p['method_target_semantics']
    out['batch_scope']={'sample_batch':1,'DT_endpoint_batch':2,'real_multi_example_batch_measured':False,
        'note':'Multiple regression examples are separate native B1 workloads. EndpointsB2 are not a two-example minibatch; no throughput claim from these serial cases.'}
    out['audit_scope']='Original formulas, own FP32 density and masks, tie-aware original needle, source/input/receipt/count checks only. No GPU/model/tokenizer replay or FT/framework edits. Audit completion is not a quality-win criterion.'
    out['cost_scope']=r['cost_scope'];out['audit_seconds']=time.perf_counter()-started
    return out

def aggregate(summaries,pending):
    cases={};counts={};sources=[]
    for s in summaries:
        for key,case in s['cases'].items():assert key not in cases;cases[key]=case
        for key,n in s['counts'].items():counts[key]=counts.get(key,0)+n
        sources.append({'segment':s['segment_index'],'results_sha256':s['results_sha256'],'protocol_sha256':s['protocol_sha256']})
    out={'status':'all_eight_cases_audited' if len(cases)==8 and not pending else 'partial_fixed_regression',
        'completed_cases':list(cases),'pending_or_failed_segments':pending,'sources':sources,'actual_completed_counts':counts,
        'datasets':{},'cases':cases,'costs_by_segment':[{'segment':s['segment_index'],'wall_seconds':s['wall_seconds'],'DT':s['DT_costs'],'FT':s['FT_costs']} for s in summaries],
        'scope':'Separate dataset macro means and within-case paired deltas. No cross-dataset victory scalar; all individual gains and regressions retained. Eight used developer cases are not independent heldout or full benchmark. Serial B1 cost is not real minibatch throughput.'}
    for dataset in ('niah_mq_q2','morehopqa'):
        selected={k:v for k,v in cases.items() if k.startswith(dataset+'_')}
        if not selected:continue
        block={'case_count':len(selected),'cases':list(selected),'method_means':{},'paired':{}}
        for m in METHODS:
            block['method_means'][m]={metric:float(np.mean([v['methods'][m][metric] for v in selected.values()])) for metric in ('RISE','MAS','alignment_augmented_AUC')}
            needles=[v['methods'][m]['needle'] for v in selected.values() if v['methods'][m]['needle'] is not None]
            block['method_means'][m]['needle_macro']=float(np.mean([x['reported'] for x in needles])) if needles else None
            block['method_means'][m]['needle_pooled_hits']=sum(x['hits'] for x in needles) if needles else None
            block['method_means'][m]['needle_pooled_gold']=sum(x['eligible_gold_denominator'] for x in needles) if needles else None
            block['method_means'][m]['MAS_decomposition']={part:float(np.mean([v['methods'][m]['MAS_decomposition'][part] for v in selected.values()]))
                for part in ('RISE','AP_AUC','negative_density_AP_extra_AUC','AP_with_nonnegative_density_AUC','original_postprocessing_change_AUC')}
        for ref in ('DT_control','FT0','FT3'):
            block['paired'][ref]={}
            for metric in ('RISE','MAS','needle'):
                values={k:v['candidate_minus_comparators'][ref][metric] for k,v in selected.items() if v['candidate_minus_comparators'][ref][metric] is not None}
                if not values:continue
                better=lambda x:x>0 if metric=='needle' else x<0
                block['paired'][ref][metric]={'mean_candidate_minus_reference':float(np.mean(list(values.values()))),
                    'gains':[k for k,x in values.items() if better(x)],'regressions':[k for k,x in values.items() if x!=0 and not better(x)],
                    'ties':[k for k,x in values.items() if x==0],'all_case_deltas':values}
        out['datasets'][dataset]=block
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--standalone-only',action='store_true');ap.add_argument('--segments-only',action='store_true');args=ap.parse_args()
    outputs=[]
    if not args.segments_only:
        D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_pristine_FT0_comparison_20260909_v1'
        s=audit_directory(D);target=A/'dt_MH0_pristine_FT0_comparison_summary_20260909.json'
        target.write_text(json.dumps(s,indent=2,allow_nan=False));outputs.append({'path':str(target),'sha256':sha(target),'deltas':{k:v['candidate_minus_comparators'] for k,v in s['cases'].items()}})
    if not args.standalone_only:
        summaries=[];pending=[]
        for i in range(4):
            D=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{i}_v1'
            if not (D/'results.json').exists() or not (D/'terminal_receipt.json').exists():
                pending.append({'segment':i,'status':'not_yet_locally_complete'});continue
            raw=read(D/'results.json')
            if raw['status']!='fixed_eight_case_regression_shard_2cases4DT2Both168score_complete':
                pending.append({'segment':i,'status':raw['status'],'error':raw.get('error'),'results_sha256':sha(D/'results.json'),
                    'known_counts':{k:raw.get(k) for k in ('DT_entered','DT_returned','FT_entered','FT_calls','scorer_entered','scorer_returned')}});continue
            s=audit_directory(D);target=A/f'dt_fixed_eight_case_regression_s{i}_summary_20260909.json';target.write_text(json.dumps(s,indent=2,allow_nan=False));summaries.append(s)
        out=aggregate(summaries,pending);out['analyzer_sha256']=sha(__file__)
        target=A/'dt_fixed_eight_case_regression_summary_20260909.json';target.write_text(json.dumps(out,indent=2,allow_nan=False))
        outputs.append({'path':str(target),'sha256':sha(target),'status':out['status'],'completed_cases':out['completed_cases'],'datasets':out['datasets']})
    print(json.dumps(outputs,ensure_ascii=False))

if __name__=='__main__':main()
