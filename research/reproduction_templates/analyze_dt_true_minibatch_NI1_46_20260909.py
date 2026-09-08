"""Independent CPU audit of the frozen real NI1+46 minibatch study.

No torch, tokenizer, model or scoring calls. A failed study retains its actual
entered/returned counts and cannot acquire a successful batch/cost verdict.
"""
import argparse, hashlib, json, time, zipfile
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import close, stats, needle, masks

A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
read=lambda f:json.loads(Path(f).read_bytes())
local=lambda s:A/'snapshot'/str(s).lstrip('/')
SUCCESS='true_distinct_NI1_46_batch2_9DT225FLA_no_scores_complete'

def drift(now,reference):
    a=np.asarray(now,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d=a-b;den=float(np.linalg.norm(b))
    return {'bitwise_equal':bool(np.array_equal(a,b)),'relative_L2':float(np.linalg.norm(d)/den) if den else None,
        'max_absolute':float(np.abs(d).max(initial=0)),'signed_sum_difference':float(d.sum())}

def summary_equal(actual,expected):
    # Local/remote CPU norm reductions need not have identical final bits.
    # This tolerance audits the reported statistic, not attribution drift.
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key,value in expected.items():summary_equal(actual[key],value)
    elif isinstance(expected,float):close(actual,expected,1e-12)
    else:assert actual==expected,(actual,expected)

def source_audit(D,p,r):
    for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
    receipt_path=D/'terminal_receipt.json';receipt=read(receipt_path) if receipt_path.exists() else None
    if receipt is not None:
        for key in ('pid_alive','proc_exists'):
            if key in receipt:assert receipt[key] is False
        for name,row in receipt['files'].items():
            assert sha(D/name)==(row if isinstance(row,str) else row['sha256']),name
            if isinstance(row,dict) and 'bytes' in row:assert (D/name).stat().st_size==row['bytes']
    if (D/'review_bundle.zip').exists():
        with zipfile.ZipFile(D/'review_bundle.zip') as z:
            for name in z.namelist():assert z.read(name)==(D/name).read_bytes(),name
    before=r.get('sources_before')
    if before is not None:
        for name,want in p['files_sha256'].items():assert before[name]==want
        for name,want in p['official_source_blob_sha1'].items():
            raw=(local(p['official_root'])/name).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want
            assert before['official/'+name]==digest(raw)
        for name,want in p['runtime_source_sha256'].items():assert before['native/'+name]==want
        for row in p['protected_sources']:assert before[row['path']]==row['sha256']
        for name,want in p['span_source_sha256'].items():
            raw=(local(p['author_data_root'])/name).read_bytes().replace(b'\r\n',b'\n')
            assert digest(raw)==want==before['author_spans/'+name]
    if 'sources_after' in r:assert r['sources_after']==before
    for key in ('weight_stats_before','weight_stats_after'):
        if key in r:assert r[key]==p['expected_weight_stats']
    return {'local_frozen_sources_verified':len(p['files_sha256']),
        'terminal_receipt_sha256':sha(receipt_path) if receipt is not None else None,
        'source_before_receipt_available':before is not None,'source_after_receipt_available':'sources_after' in r,
        'weight_after_receipt_available':'weight_stats_after' in r,
        'native_identity_scope':'Pinned native source and weight-stat receipts; no native or weight execution/reload.'}

def input_audit(p,r):
    out={};frozen=r.get('input_freeze_before_model_load',{})
    for key,f in frozen.items():
        dataset,index=key.rsplit('_',1);spec=p['fixed_records'][key]
        raw=local(p['cache_paths'][dataset]).read_bytes();assert digest(raw)==p['cache_hashes'][dataset]
        line=raw.decode().splitlines()[int(index)];record=json.loads(line)
        assert digest(line.encode())==spec['source_record_sha256']==f['mapping']['source_record_sha256']
        info=f['input'];ids=np.asarray(f['input_ids'],dtype=np.int64);target=np.asarray(f['target_ids'],dtype=np.int64)
        assert ids.shape==(info['total_length'],) and target.shape==(info['target_length'],)
        assert np.array_equal(ids[info['prompt_length']:],target) and digest(ids.tobytes())==info['input_sha256']
        base=ids.copy();base[info['keep']]=target[-1]
        assert digest(base.tobytes())==f['baseline_sha256'] and np.flatnonzero(base!=ids).tolist()==info['keep']
        m=f['mapping'];assert m['original_cached_spans_reproduced'] is True and m['original_sink_span']==record['sink_span']
        assert m['gold_function']=='unchanged author ruler_gold_prompt_token_indices'
        if spec.get('expected_input') is not None:assert info==spec['expected_input']
        if spec.get('expected_lengths') is not None:assert {k:info[k] for k in spec['expected_lengths']}==spec['expected_lengths']
        if spec.get('expected_gold') is not None:assert m['gold']==spec['expected_gold']
        if key in r['cases']:
            case=r['cases'][key];assert case['input']==info and case['mapping']==m and case['gold']==m['gold']
            assert case['baseline_sha256']==f['baseline_sha256'] and case['curves']=={}
        out[key]={'input_sha256':info['input_sha256'],'source_record_sha256':spec['source_record_sha256'],
            **{k:info[k] for k in ('total_length','prompt_length','target_length')},'gold':m['gold'],
            'eligible_count':len(info['keep']),'role':'eager_initialization_only' if key.endswith('_0') else 'true_batch_sample'}
    if set(p['batch_cases'])<=set(out):
        x,y=(out[k] for k in p['batch_cases']);assert x['input_sha256']!=y['input_sha256']
        assert x['source_record_sha256']!=y['source_record_sha256']
        assert [x[k] for k in ('total_length','prompt_length','target_length')]==[1201,937,264]
        assert [y[k] for k in ('total_length','prompt_length','target_length')]==[1201,937,264]
    return out

def audit(D):
    tick=time.perf_counter();D=Path(D);p=read(D/'protocol.json');r=read(D/'results.json')
    assert p==read(A/'dt_true_minibatch_NI1_46_protocol_20260909.json')==r['protocol']
    success=r['status']==SUCCESS
    out={'status':'independent_true_minibatch_audit_passed' if success else 'independent_failed_minibatch_partial_audit',
        'study_status':r['status'],'study_error':r.get('error'),'results_sha256':sha(D/'results.json'),
        'protocol_sha256':sha(D/'protocol.json'),'analyzer_sha256':sha(__file__),
        'shared_CPU_helper_sha256':sha(A/'analyze_dt_original_regression_20260909.py'),
        'sources':source_audit(D,p,r),'input_provenance':input_audit(p,r),'actual_seconds':r['seconds'],
        'frozen_budget':p['budget'],'runs':[],'groups':[],'per_sample_comparisons':[]}
    for k in ('scorer_entered','scorer_returned','FT_calls','generation_calls'):assert r[k]==0
    assert 0<=r['DT_returned']<=r['DT_entered']<=9
    assert r['model_loads']<=1 and r['native_eager_diagnostics']<=1
    if success:assert r['DT_returned']==9 and r['model_loads']==r['native_eager_diagnostics']==1
    vc=D/'vectors.npz';vectors=np.load(vc,allow_pickle=False) if vc.exists() else None
    out['vectors_sha256']=sha(vc) if vectors is not None else None
    if 'vectors_sha256' in r:assert sha(vc)==r['vectors_sha256']
    finite=r.get('finite_counts',{});out['actual_counts']={k:r[k] for k in ('model_loads','native_eager_diagnostics','DT_entered','DT_returned','scorer_entered','scorer_returned','FT_calls','generation_calls')}
    out['actual_counts']['finite_backend_receipts']=finite
    for name,limit in [('current',72),('FLA_backend',225),('endpoint_average_wrapper',9)]:
        if name in finite:
            c=finite[name];assert 0<=c['returned']<=c['entered']<=limit
            if success:assert c['entered']==c['returned']==limit
    if 'FLA_backend' in finite:
        c=finite['FLA_backend'];assert c['native_stages_from_returned_calls']==2*c['returned']
        assert c['inflight_native_stages']==('unknown' if c['entered']!=c['returned'] else 0)
    samples=[];returned_details=0;available_vector_keys=set()
    for run in r['runs']:
        number=run['number'];assert number==len(out['runs']);b=run['example_batch_size']
        assert b==len(run['samples'])==len(set(run['samples'])) and run['physical_endpoint_batch_size']==2*b
        assert run['samples']==(p['batch_cases'] if run['mode']=='batch2' else [p['batch_cases'][number in (1,4,8)]])
        g=p['group_schedule'][run['group']];assert [run['mode'],run['phase']]==g
        row={k:run[k] for k in ('number','group','mode','phase','samples','status','example_batch_size','physical_endpoint_batch_size','root_forwards')}
        for k in ('finite_callback_counts','native_stage_accounting','outer_attribute_seconds','GPU_allocated_before_pair','GPU_allocated_before_attribute','GPU_reserved_before_attribute','GPU_allocated_after_cleanup','GPU_reserved_after_cleanup','memory_cost'):
            if k in run:row[k]=run[k]
        row['native_runner_ledger_available']='details' in run;row['sample_records']=[]
        if 'root_endpoint_input_sha256' in run:
            expected=[value for key in run['samples'] for value in (r['input_freeze_before_model_load'][key]['baseline_sha256'],r['input_freeze_before_model_load'][key]['input']['input_sha256'])]
            assert run['root_endpoint_input_sha256']==expected
        if 'details' not in run:
            assert not success;row['unreturned_native_replay_auxiliary_and_conv_counts']='unknown'
            out['runs'].append(row);continue
        d=run['details'];returned_details+=1;assert run['root_forwards']==1
        c=run['finite_callback_counts'];assert c=={'FA_entered':8,'FA_returned':8,'FLA_entered':25,'FLA_returned':25}
        assert run['native_stage_accounting']=={'returned_backend_times_two':50,'nonreturned_backend_stage_count':0}
        assert d['norm_gate_rules']=={'0':'symmetric'} and d['finite_fla_by_layer']==[0] and d['select_output_rows'] is True
        assert d['actual_head_input_shapes']==[[2*b,264,4096]]
        assert list(d['layers'])==[str(i) for i in reversed(range(32))]
        kinds=[x['kind'] for x in d['calls']]
        for prefix in ('native_replay_','finite_decoder_'):assert [k for k in kinds if k.startswith(prefix)]==[prefix+str(i) for i in reversed(range(32))]
        assert [k for k in kinds if k.startswith('public_FA_LSE_')]==['public_FA_LSE_'+str(i) for i in reversed(p['expected_FA_layers'])]
        for layer in d['layers'].values():
            assert layer['decoder_calls']==dict.fromkeys(('input_norm','post_norm','gate','silu','up','down','mlp','decoder'),1)
            assert layer['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if layer['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
        wraps=run['wrapper_receipts'];assert len(wraps)==1 and wraps[0]['endpoint_permutation']==[i^1 for i in range(2*b)]
        assert wraps[0]['status']=='returned' and [x['orientation'] for x in wraps[0]['backend_calls']]==['original','swapped']
        assert all(x['status']=='returned' for x in wraps[0]['backend_calls'])
        if 'memory_cost' in run:
            expected={'peak_allocated':d['peak_allocated'],'peak_reserved':d['peak_reserved'],
                'peak_minus_before_pair':d['peak_allocated']-run['GPU_allocated_before_pair'],
                'peak_minus_before_attribute':d['peak_allocated']-run['GPU_allocated_before_attribute']}
            assert run['memory_cost']==expected
        offset=0
        for s in run['sample_records']:
            key=s['case'];f=r['input_freeze_before_model_load'][key];info=f['input'];vk=s['vector_key'];n=info['target_length']
            assert key==run['samples'][len(row['sample_records'])]
            close(s['root_logp0'],d['target_logp0'][offset:offset+n]);close(s['root_logp1'],d['target_logp1'][offset:offset+n]);offset+=n
            root=float((np.asarray(s['root_logp1'])-np.asarray(s['root_logp0'])).sum());close(root,s['root_effect'])
            sr={'case':key,'root_effect':root,'native_target_logp0_sum':float(np.sum(s['root_logp0'])),'native_target_logp1_sum':float(np.sum(s['root_logp1']))}
            if vectors is None or vk+'_full' not in vectors:
                assert not success;sr['vector_not_persisted']=True;row['sample_records'].append(sr);continue
            full=vectors[vk+'_full'];w=vectors[vk+'_evaluated'];available_vector_keys.update((vk+'_full',vk+'_evaluated'))
            assert full.dtype==np.float64 and w.dtype==np.float32 and full.shape==(1201,) and w.shape==(937,)
            assert np.array_equal(full[:937].astype(np.float32),w)
            for name,value in [('signed_sum',float(full.sum())),('positive',stats(full)['positive']),('negative',stats(full)['negative'])]:close(s[name],value,1e-7)
            recovered=needle(w,info['keep'],f['mapping']['gold'],s['needle'])
            masks(w,info,np.asarray(f['input_ids'],dtype=np.int64),f['target_ids'][-1],s['deletion_audit'])
            sr.update(full_signed=stats(full),eligible_signed=stats(w[info['keep']]),response_signed=stats(full[937:]),needle=recovered,
                signed_sum_minus_native_root_effect=float(full.sum()-root))
            row['sample_records'].append(sr);samples.append((run,s))
        if len(run['sample_records'])==b:
            assert offset==len(d['target_logp0'])==len(d['target_logp1'])
            close(sum(s['root_effect'] for s in run['sample_records']),d['root_effect'],1e-7)
            close(sum(s['signed_sum'] for s in run['sample_records']),d['signed_sum'],1e-7)
        out['runs'].append(row)
    assert returned_details==r['DT_returned']
    out['actual_counts'].update(native_roots_observed=sum(x['root_forwards'] for x in r['runs']),
        sample_attributions_from_returned_calls=sum(x['example_batch_size'] for x in r['runs'] if 'details' in x),
        physical_endpoint_rows_observed=sum(x['physical_endpoint_batch_size']*x['root_forwards'] for x in r['runs']),
        native_replays_from_returned_ledgers=32*returned_details,finite_decoders_from_returned_ledgers=32*returned_details,
        auxiliary_FA_from_returned_ledgers=8*returned_details,
        unreturned_native_runner_count_scope='unknown' if r['DT_entered']!=r['DT_returned'] else 'none')
    if success:assert len(samples)==12 and set(vectors.files)==available_vector_keys
    for run,s in samples:
        references=[ref for rr,ref in samples if rr['phase']=='warm' and rr['mode']=='single_pair' and ref['case']==s['case']]
        if not references:continue
        ref=references[0];key=s['case'];keep=r['input_freeze_before_model_load'][key]['input']['keep']
        w=vectors[s['vector_key']+'_evaluated'];wr=vectors[ref['vector_key']+'_evaluated'];changed=np.sign(w[keep])!=np.sign(wr[keep])
        order=s['deletion_audit']['sorted_keep'];old=ref['deletion_audit']['sorted_keep'];rank={t:i for i,t in enumerate(order)};oldrank={t:i for i,t in enumerate(old)}
        item={'run':run['number'],'mode':run['mode'],'phase':run['phase'],'case':key,
            'full':drift(vectors[s['vector_key']+'_full'],vectors[ref['vector_key']+'_full']),'evaluated':drift(w,wr),
            'root_logp0':drift(s['root_logp0'],ref['root_logp0']),'root_logp1':drift(s['root_logp1'],ref['root_logp1']),
            'root_effect_difference':s['root_effect']-ref['root_effect'],'needle':s['needle'],
            'needle_minus_single':None if s['needle'] is None else s['needle']-ref['needle'],
            'eligible_sign_changes':int(changed.sum()),'reference_absolute_mass_on_sign_changes':float(np.abs(wr[keep][changed].astype(np.float64)).sum()),
            'sorted_keep_equal':order==old,'maximum_rank_displacement':max(abs(rank[k]-oldrank[k]) for k in keep),
            'same_deletion_input_count':sum(a==b for a,b in zip(s['deletion_audit']['input_receipts'],ref['deletion_audit']['input_receipts']))}
        if 'per_sample_comparisons' in r:
            original=next(x for x in r['per_sample_comparisons'] if x['run']==run['number'] and x['case']==key)
            for k,v in item.items():summary_equal(original[k],v)
        out['per_sample_comparisons'].append(item)
    for g in r.get('groups',[]):
        row=dict(g);out['groups'].append(row)
        if g['status']!='complete':continue
        expected=[x['number'] for x in r['runs'] if x['group']==g['number']]
        assert g['invocations']==expected and len(expected)==(2 if g['mode']=='single_pair' else 1)
        runs=[r['runs'][i] for i in expected];assert sum(x['example_batch_size'] for x in runs)==2
        close(g['attribute_seconds_sum'],sum(x['outer_attribute_seconds'] for x in runs))
        close(g['runner_seconds_sum'],sum(x['details']['complete_attribution_seconds_with_diagnostics'] for x in runs))
        for k in ('peak_allocated','peak_reserved'):assert g[k]==max(x['details'][k] for x in runs)
        row['examples']=2;row['seconds_per_example']=g['attribute_seconds_sum']/2
    out['cost_summary']={}
    if success:
        assert len(out['groups'])==6 and [x['status'] for x in out['groups']]==['complete']*6
        for mode,want in [('single_pair',[2,5]),('batch2',[3,4])]:
            groups=[g for g in out['groups'] if g['phase']=='measured' and g['mode']==mode]
            assert [g['number'] for g in groups]==want
            t=[g['attribute_seconds_sum'] for g in groups];m=float(np.median(t));saved=r['cost_summary'][mode]
            assert saved['seconds_for_two_examples']==t and saved['groups']==want
            close(saved['median_seconds_for_two_examples'],m);close(saved['mean_seconds_for_two_examples'],float(np.mean(t)))
            close(saved['median_examples_per_second'],2/m)
            for k in ('peak_allocated','peak_reserved'):assert saved[k]==[g[k] for g in groups]
            out['cost_summary'][mode]={**saved,'median_seconds_per_example':m/2,'mean_examples_per_second_across_observations':float(np.mean(2/np.asarray(t)))}
        cost=out['cost_summary'];ratio=cost['batch2']['median_seconds_for_two_examples']/cost['single_pair']['median_seconds_for_two_examples']
        close(ratio,r['cost_ratio_batch_over_sequential_pair']);out['batch_over_sequential_two_example_time_ratio']=ratio
        out['batch_over_sequential_example_throughput_ratio']=1/ratio
    out['scope']='True distinct NI1+46, B2 examples/B4 endpoints versus the same two sequential singletons. No padding or duplicate examples. Per-example target, signed/rank/needle drift is reported without an arbitrary precision pass threshold; no RISE/MAS measured. Only two warm-excluded S/B/B/S observations per mode support the fixed-length cost comparison. Failed runs have no completed cost claim.'
    out['study_cost_scope']=r.get('cost_scope');out['audit_seconds']=time.perf_counter()-tick
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',nargs='?',default=str(A/'snapshot${ARTIFACT_ROOT}/codex_dt_true_minibatch_NI1_46_20260909_v1'));args=ap.parse_args()
    result=audit(args.directory);target=A/'dt_true_minibatch_NI1_46_summary_20260909.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'sha256':sha(target),'status':result['status'],
        'actual_seconds':result['actual_seconds'],'cost_summary':result['cost_summary']},ensure_ascii=False))
