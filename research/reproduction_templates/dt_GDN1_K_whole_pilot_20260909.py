"""One fixed remaining-case input-supported GDN1 K pilot;4DT100FLA84originalscores."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_entered':0,'DT_returned':0,
   'scorer_entered':0,'scorer_returned':0,'FT_calls':0,'generation_calls':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();handles=[];vectors={};active=None
fas={};finite_fla=None;candidate_backend=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen4DT100FLA32FA84score input-supported GDN1 K budget expired.')
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        out=fn();torch.cuda.synchronize();row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-tick
def sources():
    found={}
    for name,want in p['files_sha256'].items():
        found[name]=sha((A/name).read_bytes());assert found[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        found['official/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        found['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert found['native/'+name]==want
    for name,want in p['span_source_sha256'].items():
        found['author_spans/'+name]=sha((Path(p['author_data_root'])/name).read_bytes().replace(b'\r\n',b'\n'));assert found['author_spans/'+name]==want
    for item in p['protected_sources']:
        found[item['path']]=sha(Path(item['path']).read_bytes());assert found[item['path']]==item['sha256']
    return found

try:
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from official_span_mapping import load_author_span_helpers
    from fixed_input_metric_view import FixedInputMetricView
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from layer0_fla_endpoint_average_20260908 import Layer0FLAEndpointAverage
    from normal_finite_study_utils_20260908 import CountFinite,check_counts,deletion_audit
    class BudgetedFinite(CountFinite):
        def __init__(self,operation,limit):super().__init__(operation);self.limit=limit
        def __call__(self,*args,**kwargs):
            assert self.entered<self.limit,'Frozen finite-backend budget exhausted.'
            return super().__call__(*args,**kwargs)
    remaining_pairs=[['niah_mq_q2_0','morehopqa_2']]
    assert type(p['segment_index']) is int and p['segment_index']==0
    first,second=remaining_pairs[p['segment_index']]
    assert p['quality_cases']==[first,second]
    assert p['call_schedule']==[[first,'control','quality'],[first,'candidate','quality'],[second,'candidate','quality'],[second,'control','quality']]
    assert p['quality_schedule']==[row[:2] for row in p['call_schedule']]
    expected_cases=list(dict.fromkeys(['niah_mq_q2_0',first,second]))
    assert p['case_indices']==[[key.rsplit('_',1)[0],int(key.rsplit('_',1)[1])] for key in expected_cases]
    r['segment_index']=p['segment_index']
    r['completion_scope']='Only input-supported GDN1 K NI0/MH2 pilot; no other cases dispatched or claimed complete for this candidate.'
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    old_tokenizer=AutoTokenizer.from_pretrained(p['old_checkpoint'],local_files_only=True);old_tokenizer.pad_token=old_tokenizer.eos_token
    helpers=load_author_span_helpers(p['author_data_root'],p['span_source_sha256'])
    data={name:Path(path).read_bytes() for name,path in p['cache_paths'].items()}
    for name,raw in data.items():assert sha(raw)==p['cache_hashes'][name]
    def prepare(dataset,index):
        line=data[dataset].decode().splitlines()[index];rec=json.loads(line);key=f'{dataset}_{index}'
        assert sha(line.encode())==p['fixed_records'][key]['source_record_sha256']
        cls=helpers['CachedExample'];fields=['prompt','target','indices_to_explain','attr_mask_indices','sink_span','thinking_span','metadata']
        cached=cls(**{name:rec.get(name) for name in fields})
        def remap(tok):
            fresh=cls(prompt=cached.prompt,target=cached.target,indices_to_explain=None,attr_mask_indices=cached.attr_mask_indices,
                sink_span=None,thinking_span=None,metadata=dict(cached.metadata))
            fresh=helpers['attach_spans_from_answer'](fresh,tok);fresh.indices_to_explain=list(fresh.sink_span);return fresh
        old,new=remap(old_tokenizer),remap(tokenizer)
        for name in ('sink_span','thinking_span','indices_to_explain'):assert getattr(old,name)==getattr(cached,name),name
        tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        labels=[rec['prompt'][a:b] for a,b in tok['offset_mapping']]
        keep=keep_token_indices(labels);assert keep==helpers['keep_token_indices'](labels)
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        assert target==tokenizer(rec['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        assert len(ids)<=p['max_total_tokens'] and len(keep)>=20 and (base!=ids).nonzero().flatten().tolist()==keep
        gold=helpers['ruler_gold_prompt_token_indices'](new,tokenizer)
        info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),'target_length':len(target),'total_length':len(ids),'keep':keep}
        ref=p['fixed_records'][key]
        if ref.get('expected_input') is not None:assert info==ref['expected_input']
        if ref.get('expected_gold') is not None:assert gold==ref['expected_gold']
        mapping={'source_record_sha256':sha(line.encode()),'original_cached_spans_reproduced':True,
            'original_sink_span':old.sink_span,'new_sink_span':new.sink_span,'new_thinking_span':new.thinking_span,
            'gold':gold,'gold_function':'unchanged author ruler_gold_prompt_token_indices',
            'input_scope':'Original author prompt plus fixed original target and EOS;no additional chat template;same wholepilot pipeline.',
            'historical_input_hash_available':ref.get('expected_input') is not None}
        return {'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info,'gold':gold,'mapping':mapping}
    cases={f'{dataset}_{index}':prepare(dataset,index) for dataset,index in p['case_indices']}
    init=cases['niah_mq_q2_0']
    r['input_freeze_before_model_load']={key:dict(input=case['input'],mapping=case['mapping'],
        input_ids=case['ids'].tolist(),target_ids=case['target'].tolist(),baseline_sha256=sha(case['base'].numpy().tobytes())) for key,case in cases.items()}
    save()
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    layers=model.model.language_model.layers
    assert [i for i,layer in enumerate(layers) if layer.block_type=='full_attention']==p['expected_FA_layers']
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_NI0_eager_initialization',lambda:model(input_ids=init['ids'][None].to('cuda'),
        attention_mask=torch.ones_like(init['ids'][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,init;model.set_attn_implementation('flash_attention_2')
    original_fa=VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256'])
    fas={method:BudgetedFinite(original_fa,16) for method in ('control','candidate')}
    assert fas['control'].operation is fas['candidate'].operation is original_fa
    finite_fla=BudgetedFinite(make_compiled_finite_pullback(reuse_scalar_products=False),100)
    candidate_backend=Layer0FLAEndpointAverage(finite_fla)
    from input_supported_l2_20260909 import input_supported_l2_pullback
    key_calls=[]
    def key_candidate(x0,x1,upstream):
        assert len(key_calls)<2
        key_calls.append({'input_shape':list(x0.shape),'upstream_shape':list(upstream.shape),'status':'entered'})
        result=input_supported_l2_pullback(x0.float(),x1.float(),upstream,eps=1e-6)
        key_calls[-1]['status']='returned'
        return result
    runners={method:Qwen35DenseFiniteRunner(model,backend,finite_fla,norm_gate_rules={0:'symmetric'},
        finite_fla_by_layer={0:candidate_backend},
        key_norm_by_layer=({1:key_candidate} if method=='candidate' else {})) for method,backend in fas.items()}
    assert runners['control'].key_norm_by_layer=={} and runners['candidate'].key_norm_by_layer=={1:key_candidate}
    assert all(runner.attention_pv_rules=={} for runner in runners.values())
    assert all(runner.finite_fla_by_layer=={0:candidate_backend} for runner in runners.values())
    assert all(runner.finite_fla is finite_fla for runner in runners.values())
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key,case in cases.items():
        if key not in p['quality_cases']:continue
        r['cases'][key]={'input':case['input'],'gold':case['gold'],'mapping':case['mapping'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},
            'role':'fixed_used_author_case_original_quality',
            'historical_FT_reference':p['historical_FT_references'][key]}
    for number,(key,method,phase) in enumerate(p['call_schedule']):
        gc.collect();torch.cuda.synchronize();resident_before_pair=torch.cuda.memory_allocated();reserved_before_pair=torch.cuda.memory_reserved()
        case=cases[key];info=case['input'];pairs=torch.stack((case['base'],case['ids'])).to('cuda')
        selection=PackedAnswerTargets([{'target_ids':case['target'],'prompt_length':info['prompt_length']}],
            [list(range(len(case['target'])))],len(case['ids']),'cuda')
        row={'case':key,'method':method,'phase':phase,'number':number,'status':'entered','root_forwards':0,
            'GPU_allocated_before_pair':resident_before_pair,'GPU_reserved_before_pair':reserved_before_pair}
        r['runs'].append(row);r['status']='attribute_'+key+'_'+method;save()
        def before_root(_module,args,kw):
            assert torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
            row['root_forwards']+=1
        handle=model.register_forward_pre_hook(before_root,with_kwargs=True)
        old_fa=fas[method].returned;old_fla=finite_fla.returned;old_receipts=len(candidate_backend.calls)
        old_fa_entered=fas[method].entered;old_fla_entered=finite_fla.entered
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_before_attribute']=torch.cuda.memory_allocated();row['GPU_reserved_before_attribute']=torch.cuda.memory_reserved();assert r['DT_entered']<4;r['DT_entered']+=1;tick=time.perf_counter();details=None
        try:
            signed,details=runners[method].attribute(pairs,torch.ones_like(pairs),selection,select_output_rows=True,observer=None)
            r['DT_returned']+=1;row['details']=details
        finally:
            handle.remove()
            try:torch.cuda.synchronize()
            except Exception:row['final_sync_error']=traceback.format_exc()
            row['outer_attribute_seconds']=time.perf_counter()-tick
            row['finite_callback_counts']={'FA_entered':fas[method].entered-old_fa_entered,'FA_returned':fas[method].returned-old_fa,
                'FLA_backend_entered':finite_fla.entered-old_fla_entered,'FLA_backend_returned':finite_fla.returned-old_fla}
            row['candidate_layer0_receipts']=candidate_backend.calls[old_receipts:]
            row['native_stage_accounting']={'returned_FLA_backend_calls_times_two':2*(finite_fla.returned-old_fla),
                'stages_inside_nonreturned_backend':'unknown' if finite_fla.entered-old_fla_entered!=finite_fla.returned-old_fla else 0}
            if details is None:row['status']='failed_attribute';row['native_runner_ledger']='unknown: attribution did not return details'
            save()
        assert row['root_forwards']==1
        extra=1  # Common current layer0 endpoint average in both methods.
        # Both methods have24 GDN sites and25 actual finite FLA executions.
        fa_extra=0
        row['counts']=check_counts(details,fas[method].returned-old_fa-fa_extra,finite_fla.returned-old_fla-extra)
        row['counts']['finite_FA_backend_calls']=fas[method].returned-old_fa
        assert row['counts']['finite_FA_backend_calls']==8+fa_extra
        row['counts']['finite_FLA_backend_calls']=finite_fla.returned-old_fla
        row['counts']['finite_FLA_calls_scope']='24 original GDN callback sites; see finite_FLA_backend_calls for actual executions'
        assert row['counts']['finite_FLA_backend_calls']==24+extra
        assert details['norm_gate_rules']=={'0':'symmetric'} and details['finite_fla_by_layer']==[0]
        assert details['attention_pv_rules']=={}
        assert details['key_norm_by_layer']==([1] if method=='candidate' else [])
        row['PV_rule_scope']=details['attention_pv_rules']
        row['finite_FA_phase_accounting']={'callback_sites':8,'returned_calls':row['counts']['finite_FA_backend_calls'],'phases_from_returned_calls':3*row['counts']['finite_FA_backend_calls'],'library':p['finite_FA_library']}
        assert details['select_output_rows'] is True
        kinds=[item['kind'] for item in details['calls']]
        assert [x for x in kinds if x.startswith('native_replay_')]==['native_replay_'+str(i) for i in reversed(range(32))]
        assert [x for x in kinds if x.startswith('finite_decoder_')]==['finite_decoder_'+str(i) for i in reversed(range(32))]
        assert list(details['layers'])==[str(i) for i in reversed(range(32))]
        assert [int(i) for i,x in details['layers'].items() if x['block_type']=='linear_attention']==p['expected_GDN_reverse_order']
        assert len(row['candidate_layer0_receipts'])==extra
        assert all(x['status']=='returned' and [c['orientation'] for c in x['backend_calls']]==['original','swapped']
            and all(c['status']=='returned' for c in x['backend_calls']) for x in row['candidate_layer0_receipts'])
        signed=signed[0];score=signed[:info['prompt_length']].float();assert bool(torch.isfinite(signed).all())
        row['status']='complete'
        row['deletion_audit']=deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)
        vector_key=key+'_'+method+('_run'+str(number) if phase!='quality' else '')
        row['vector_key']=vector_key
        vectors[vector_key+'_full']=signed.numpy().copy();vectors[vector_key+'_evaluated']=score.numpy().copy()
        row['signed_summary']={'net':float(signed.double().sum()),'positive':float(signed[signed>0].double().sum()),
            'negative':float(signed[signed<0].double().sum()),'needle':float(evaluate_attr_recovery_skip_tokens(score[None],
                keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=case['gold'],top_fraction=.1)) if case['gold'] else None}
        row['memory_cost']={'peak_allocated_full_model_resident':row['details']['peak_allocated'],
            'peak_reserved_full_model_resident':row['details']['peak_reserved'],
            'peak_allocated_minus_before_pair':row['details']['peak_allocated']-resident_before_pair,
            'peak_allocated_minus_before_attribute':row['details']['peak_allocated']-row['GPU_allocated_before_attribute']}
        np.savez_compressed(A/'vectors.npz',**vectors);del signed,score,pairs,selection,details
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_after_cleanup']=torch.cuda.memory_allocated();row['GPU_reserved_after_cleanup']=torch.cuda.memory_reserved();save()
    assert r['DT_entered']==r['DT_returned']==4
    # Original scorer forwards, with own actual vectors and masks for every curve.
    for key,method in p['quality_schedule']:
        case=cases[key];info=case['input'];score=torch.from_numpy(vectors[key+'_'+method+'_evaluated'])
        run=next(x for x in r['runs'] if x['case']==key and x['method']==method)
        view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        curve={'input_receipts':[],'returned_forwards':0,'needle':float(evaluate_attr_recovery_skip_tokens(score[None],
            keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=r['cases'][key]['gold'],top_fraction=.1)) if r['cases'][key]['gold'] else None}
        r['cases'][key]['curves'][method]=curve
        def before_score(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scorer_entered']<84
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(case['ids'])) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=case['ids']).nonzero().flatten().tolist()
            receipt={'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed}
            assert receipt==run['deletion_audit']['input_receipts'][step]
            curve['input_receipts'].append(receipt);r['scorer_entered']+=1
        def after_score(_module,args,output):
            if output is not None:
                assert output.logits.dtype==torch.bfloat16;r['scorer_returned']+=1;curve['returned_forwards']+=1
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                loc=frame.f_locals
                for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                    curve[name]=np.asarray(loc[name]).copy().tolist()
                curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
        handles=[model.register_forward_pre_hook(before_score,with_kwargs=True),model.register_forward_hook(after_score,always_call=True)]
        assert sys.getprofile() is None;r['status']='original_curve_'+key+'_'+method;save();sys.setprofile(observe)
        try:
            with torch.no_grad():returned=timed('original_curve_'+key+'_'+method,lambda:faithfulness_test_skip_tokens(view,score[None],
                case['record']['prompt'],case['record']['target'],keep_prompt_token_indices=info['keep'],user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[]
        curve['return_metrics']=[float(x) for x in returned]
        assert curve['returned_forwards']==21 and curve['sorted_keep']==run['deletion_audit']['sorted_keep']
        assert all(np.isfinite(curve[name]).all() for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'))
        curve['status']='complete';save()
    assert r['scorer_entered']==r['scorer_returned']==84
    for key in p['quality_cases']:
        case=cases[key];row=r['cases'][key];control=row['curves']['control'];candidate=row['curves']['candidate']
        row['clean_and_allEOS_scores_equal_between_methods']=[control['scores'][i]==candidate['scores'][i] for i in (0,20)]
        row['fixed_control_masks']={'source':'Actual newly scored control21-point curve in this same execution;0additional scorers. Raw conditional prediction ledger,not another MAS curve.', 'points':[]}
        for step,receipt in enumerate(control['input_receipts']):
            actual=control['scores'][0]-control['scores'][step];deleted=receipt['deleted_positions']
            entry={'step':step,'input_receipt':receipt,'actual_logprob_drop':actual}
            for method in ('control','candidate'):
                value=float(vectors[key+'_'+method+'_evaluated'][deleted].astype(np.float64).sum())
                entry[method]={'deleted_signed_sum':value,'prediction_minus_actual':value-actual}
            row['fixed_control_masks']['points'].append(entry)
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    assert all(fas[method].entered==fas[method].returned==(16) for method in fas)
    assert finite_fla.entered==finite_fla.returned==100
    r['key_norm_calls']=key_calls
    assert len(key_calls)==2 and all(x['status']=='returned' for x in key_calls)
    assert candidate_backend.entered==candidate_backend.returned==4
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'] and sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='GDN1_K_input_supported_4DT100FLA84score_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    if finite_fla is not None:
        r['finite_counts']['FLA_backend']={'entered':finite_fla.entered,'returned':finite_fla.returned,
            'native_adjoint_stages_from_returned_calls':2*finite_fla.returned,
            'native_stages_inside_nonreturned_calls':'unknown' if finite_fla.entered!=finite_fla.returned else 0}
    if candidate_backend is not None:r['finite_counts']['layer0_average_wrapper']={
        'entered':candidate_backend.entered,'returned':candidate_backend.returned,'calls':candidate_backend.calls}
    r['cost_scope']='One resident native model,4DT/84originalscore.32finiteFA/96phases,100finiteFLA/200nativeadjoints,128native replays.Only2GDN1 K finite norm callbacks use the frozen input-supported rule; no native model/FA/FLA changes, extra forwards, numerical path steps or T-squared matrices.Cold/order-confounded pilot timing, not hot or true batch acceptance.'
    r['finite_counts']['original_FA_phases_from_returned_calls']=3*sum(op.returned for op in fas.values())
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_entered':r['DT_entered'],'DT_returned':r['DT_returned'],
        'scorer_entered':r['scorer_entered'],'scorer_returned':r['scorer_returned'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
