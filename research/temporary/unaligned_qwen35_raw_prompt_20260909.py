"""WITHDRAWN, NEVER EXECUTED: unaligned raw-prompt draft; not a release entrypoint."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_entered':0,'DT_returned':0,
   'scorer_entered':0,'scorer_returned':0,'FT_calls':0,'FT_entered':0,'generation_calls':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();handles=[];vectors={};active=None
fas={};finite_fla=None;candidate_backend=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen clean DT/FT evaluation budget expired.')
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        out=fn();torch.cuda.synchronize();row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-tick
def sources():
    found={}
    assert {str(f.relative_to(Path(p['official_root']))) for f in (Path(p['official_root'])/'flashtrace').rglob('*.py')}==set(p['official_package_blob_sha1'])
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
    from collections import Counter
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from flashtrace import FlashTrace
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from official_span_mapping import load_author_span_helpers
    from fixed_input_metric_view import FixedInputMetricView
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_clean_runner import make_qwen35_clean_runner
    from metric_score_view import positive_metric_scores
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from normal_finite_study_utils_20260908 import CountFinite,check_counts,deletion_audit
    class BudgetedFinite(CountFinite):
        def __init__(self,operation,limit):super().__init__(operation);self.limit=limit
        def __call__(self,*args,**kwargs):
            assert self.entered<self.limit,'Frozen finite-backend budget exhausted.'
            return super().__call__(*args,**kwargs)
    full_cases=[f'{d}_{i}' for d in ('niah_mq_q2','morehopqa') for i in range(8)]
    selected=full_cases if p['shard']=='all' else [f'{d}_{p["shard"]}' for d in ('niah_mq_q2','morehopqa')]
    assert p['quality_cases']==selected
    assert p['quality_schedule']==[[key,method] for key in selected for method in ('DT','FT0','FT3')]
    r['scope']='Clean content1/P1, no layer overrides. Positive DT metric input; signed vectors retained. Original FT unchanged.'
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
            'historical_input_hash_available':ref.get('expected_input') is not None,
            'internal_target_semantics':'FT original sink-span representation and recursive all-generation policy;DT full fixed-response logprob including EOS. Shared raw input,response and author scorer target;internal seeds are not identical.'}
        return {'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info,'gold':gold,'mapping':mapping}
    cases={f'{dataset}_{index}':prepare(dataset,index) for dataset,index in p['case_indices']}
    init=cases.get('niah_mq_q2_0')
    if init is None:init=prepare('niah_mq_q2',0)
    r['initialization_input']=init['input']
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
    fas={'DT':BudgetedFinite(original_fa,8*len(selected))}
    finite_fla=BudgetedFinite(make_compiled_finite_pullback(reuse_scalar_products=False),24*len(selected))
    runner=make_qwen35_clean_runner(model,fas['DT'],finite_fla)
    for name in ('norm_gate_rules','finite_fla_by_layer','attention_pv_rules','key_norm_by_layer'):
        assert getattr(runner,name)=={}
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'gold':case['gold'],'mapping':case['mapping'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},'role':'clean_v1_frozen_selection'}
    for number,key in enumerate(selected):
        case=cases[key];info=case['input'];gc.collect();torch.cuda.synchronize()
        before=torch.cuda.memory_allocated();tick=time.perf_counter()
        pairs=torch.stack((case['base'],case['ids'])).to('cuda')
        selection=PackedAnswerTargets([{'target_ids':case['target'],'prompt_length':info['prompt_length']}],
            [list(range(len(case['target'])))],len(case['ids']),'cuda')
        row={'case':key,'method':'DT','number':number,'status':'entered','root_forwards':0,
             'allocated_before_pair':before,'true_sample_batch':1,'endpoint_rows':2}
        r['runs'].append(row);r['status']='clean_attribute_'+key;save()
        def before_root(_module,args,kw):
            assert torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
            row['root_forwards']+=1
        handle=model.register_forward_pre_hook(before_root,with_kwargs=True)
        fa_before=fas['DT'].returned;fla_before=finite_fla.returned;r['DT_entered']+=1
        try:
            signed,details=runner.attribute(pairs,torch.ones_like(pairs),selection,select_output_rows=True,observer=None)
            r['DT_returned']+=1
        finally:handle.remove()
        assert row['root_forwards']==1
        row['counts']=check_counts(details,fas['DT'].returned-fa_before,finite_fla.returned-fla_before)
        for name in ('norm_gate_rules','attention_pv_rules'):assert details[name]=={}
        for name in ('finite_fla_by_layer','key_norm_by_layer'):assert details[name]==[]
        assert details['select_output_rows'] is True
        signed=signed[0];assert bool(torch.isfinite(signed).all())
        score=positive_metric_scores(signed[:info['prompt_length']])
        vectors[key+'_DT_full']=signed.numpy().copy()
        vectors[key+'_DT_signed_prompt']=signed[:info['prompt_length']].numpy().copy()
        vectors[key+'_DT_evaluated']=score.numpy().copy()
        row['deletion_audit']=deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)
        torch.cuda.synchronize();row['whole_attribution_seconds']=time.perf_counter()-tick
        row['details']=details;row['status']='complete'
        row['memory_cost']={'peak_allocated':details['peak_allocated'],'peak_reserved':details['peak_reserved'],
            'peak_allocated_minus_before_pair':details['peak_allocated']-before}
        np.savez_compressed(A/'vectors.npz',**vectors)
        print(json.dumps({'case':key,'stage':'clean_DT','seconds':row['whole_attribution_seconds']}),flush=True)
        del signed,score,pairs,selection,details;gc.collect();save()
    assert r['DT_entered']==r['DT_returned']==len(selected)
    def run_ft(key):
        # Entire unchanged public Both API, including all author hybrid probes.
        model.set_attn_implementation('eager')
        assert all(layer.self_attn.config._attn_implementation=='eager' for layer in layers if layer.block_type=='full_attention')
        case=cases[key];info=case['input'];ft={'status':'entered','root_entered':0,'root_returned':0,'input_receipts':[],'events':[]}
        r['cases'][key]['FT']=ft;counts=Counter();r['status']='unchanged_complete_FT_Both_'+key;save()
        def before_ft(_module,args,kw):
            ft['root_entered']+=1;assert ft['root_entered']==1
            x=kw['input_ids'].detach().cpu();assert torch.equal(x,case['ids'][None])
            assert kw.get('output_attentions') is True and kw.get('use_cache') is False and bool(kw['attention_mask'].eq(1).all())
            ft['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'shape':list(x.shape)})
        def after_ft(_module,args,kw,out):
            if out is not None:
                ft['root_returned']+=1;ft['attention_shapes']=[list(x.shape) for x in out.attentions]
                assert len(out.attentions)==8
        def observe_ft(frame,event,arg):
            if event!='call':return
            name=frame.f_code.co_name
            if name not in {'calculate_ifr_multi_hop_both','_capture_model_state','build_layer_inputs',
                '_linear_layer_input','_full_layer_input','compute_ifr_sentence_aggregate','chunk_gated_delta_rule','eager_attention_forward'}:return
            filename=frame.f_code.co_filename
            if filename.startswith(p['official_root']+'/'):counts['official.'+name]+=1
            elif name=='chunk_gated_delta_rule' and '/fla/ops/gated_delta_rule/chunk.py' in filename:
                counts['native_FLA.chunk_gated_delta_rule']+=1;v=frame.f_locals.get('v')
                if v is not None:ft['events'].append({'kind':'native_FLA_call','value_shape':list(v.shape)})
            elif name=='eager_attention_forward' and '/transformers/models/qwen3_5/' in filename:counts['native_model.eager_attention_forward']+=1
        handles=[model.register_forward_pre_hook(before_ft,with_kwargs=True),model.register_forward_hook(after_ft,with_kwargs=True)]
        tracer=FlashTrace(model,tokenizer);gc.collect();torch.cuda.synchronize();ft['allocated_before']=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        assert sys.getprofile() is None and r['FT_entered']<len(selected);r['FT_entered']+=1;tick=time.perf_counter();sys.setprofile(observe_ft)
        try:
            result=tracer.trace(prompt=case['record']['prompt'],target=case['record']['target'],
                output_span=tuple(case['mapping']['new_sink_span']),reasoning_span=tuple(case['mapping']['new_thinking_span']),hops=3,method='flashtrace')
            r['FT_calls']+=1;ft['status']='trace_returned'
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[]
            try:torch.cuda.synchronize()
            except Exception:ft['final_sync_error']=traceback.format_exc()
            ft['seconds_with_passive_counts']=time.perf_counter()-tick;ft['counts']=dict(counts)
            ft['peak_allocated']=torch.cuda.max_memory_allocated();ft['peak_reserved']=torch.cuda.max_memory_reserved();save()
            # Use official model backend setter even on failure;never patch author methods.
            model.set_attn_implementation('flash_attention_2')
        assert ft['root_entered']==ft['root_returned']==1
        assert counts['official.calculate_ifr_multi_hop_both']==1
        assert counts['official._linear_layer_input']==24 and counts['official._full_layer_input']==8
        assert counts['native_FLA.chunk_gated_delta_rule']==72 and counts['native_model.eager_attention_forward']==8
        assert counts['official.compute_ifr_sentence_aggregate']<=4
        assert len(result.prompt_tokens)==info['prompt_length'] and keep_token_indices(result.prompt_tokens)==info['keep']
        ifr=result.metadata['ifr'];obs=ifr['observation_projected'];prefix=obs['base'].clone();cumulative=[prefix.clone()]
        for extra_hop in obs['per_hop']:prefix=prefix+extra_hop;cumulative.append(prefix.clone())
        assert len(cumulative)==4 and torch.equal(prefix,obs['sum'])
        assert np.array_equal(prefix[:info['prompt_length']].cpu().numpy(),np.asarray(result.scores,dtype=np.float32))
        all_scores=torch.stack(cumulative)[:,:info['prompt_length']].detach().cpu();assert bool(torch.isfinite(all_scores).all())
        for hop in range(4):
            vectors[key+f'_FT{hop}_evaluated']=all_scores[hop].numpy().copy()
            vectors[key+f'_FT{hop}_full']=cumulative[hop].detach().cpu().numpy().copy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        ft['official_span_metadata']={name:ifr[name] for name in ['sink_span_generation','thinking_span_generation','all_gen_span_generation','n_hops','stop_config','note']}
        assert list(ifr['sink_span_generation'])==list(case['mapping']['new_sink_span'])
        assert list(ifr['thinking_span_generation'])==list(case['mapping']['new_thinking_span'])
        assert ifr['n_hops']==3
        ft['deletion_audits']={f'FT{hop}':deletion_audit(all_scores[hop],case['ids'],info['keep'],tokenizer.eos_token_id) for hop in (0,3)}
        ft['status']='complete';ft['scored_hops']=[0,3];ft['unscored_hops']=[1,2]
        del result,tracer,ifr,obs,prefix,cumulative,all_scores,extra_hop
        gc.collect();torch.cuda.synchronize();ft['allocated_after_cleanup']=torch.cuda.memory_allocated()
        assert all(layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule for layer in layers if layer.block_type=='linear_attention')
        assert all(layer.self_attn.config._attn_implementation=='flash_attention_2' for layer in layers if layer.block_type=='full_attention');save()
    for key in selected:run_ft(key)
    assert r['FT_entered']==r['FT_calls']==len(selected)
    # Original scorer forwards, with own actual vectors and masks for every curve.
    for key,method in p['quality_schedule']:
        case=cases[key];info=case['input'];score=torch.from_numpy(vectors[key+'_'+method+'_evaluated'])
        run={'deletion_audit':r['cases'][key]['FT']['deletion_audits'][method]} if method.startswith('FT') else next(x for x in r['runs'] if x['case']==key and x['method']==method)
        view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        curve={'input_receipts':[],'returned_forwards':0,'needle':float(evaluate_attr_recovery_skip_tokens(score[None],
            keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=r['cases'][key]['gold'],top_fraction=.1)) if r['cases'][key]['gold'] else None}
        r['cases'][key]['curves'][method]=curve
        def before_score(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scorer_entered']<p['budget']['scoring_forwards']
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
    assert r['scorer_entered']==r['scorer_returned']==p['budget']['scoring_forwards']
    assert fas['DT'].returned==8*len(selected) and finite_fla.returned==24*len(selected)
    for key in selected:
        row=r['cases'][key];curves=row['curves']
        row['DT_minus_FT']={name:{'metrics':[a-b for a,b in zip(curves['DT']['return_metrics'],curves[name]['return_metrics'])],
            'needle':curves['DT']['needle']-curves[name]['needle'] if curves[name]['needle'] is not None else None} for name in ('FT0','FT3')}
        row['endpoint_scores']={name:[curve['scores'][i] for i in (0,20)] for name,curve in curves.items()}
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='clean_v1_selected_cases_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    if finite_fla is not None:r['finite_counts']['FLA']={'entered':finite_fla.entered,'returned':finite_fla.returned,
        'native_adjoint_stages_from_returned_calls':2*finite_fla.returned}
    r['cost_scope']='One model load and original NI0 initialization; each selected case has one clean DT, one entire unchanged FT Both hops3 and three original 21-score curves. Cold compile/diagnostics are included; this is B1, not a hot throughput or true batch claim.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'DT':r['DT_returned'],'FT':r['FT_calls'],
        'scores':r['scorer_returned'],'error':r.get('error')}),flush=True)
