"""Four unchanged original curves for saved singleton and real-batch NI vectors; no DT."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_entered':0,'DT_returned':0,'finite_FA_calls':0,'finite_FLA_calls':0,
   'scorer_entered':0,'scorer_returned':0,'FT_calls':0,'generation_calls':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();handles=[];vectors={};active=None
fas={};finite_fla=None;candidate_backend=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen84-original-score saved minibatch-vector budget expired.')
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
    from fixed_input_metric_view import FixedInputMetricView
    from official_span_mapping import load_author_span_helpers
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    old_tokenizer=AutoTokenizer.from_pretrained(p['old_checkpoint'],local_files_only=True);old_tokenizer.pad_token=old_tokenizer.eos_token
    helpers=load_author_span_helpers(p['author_data_root'],p['span_source_sha256'])
    data={name:Path(path).read_bytes() for name,path in p['cache_paths'].items()}
    for name,raw in data.items():assert sha(raw)==p['cache_hashes'][name]
    source=p['source_batch']
    for name in ['results','vectors','protocol']:
        assert sha(Path(source[name+'_path']).read_bytes())==source[name+'_sha256']
    prior=json.loads(Path(source['results_path']).read_bytes());source_protocol=json.loads(Path(source['protocol_path']).read_bytes())
    assert prior['protocol']==source_protocol and prior['status']=='true_distinct_NI1_46_batch2_9DT225FLA_no_scores_complete'
    for name in p['same_native_identity_keys']:assert p[name]==source_protocol[name],name
    with np.load(source['vectors_path'],allow_pickle=False) as archive:
        saved_vectors={name:archive[name].copy() for name in archive.files}
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
        if ref.get('expected_lengths') is not None:assert {k:info[k] for k in ['total_length','prompt_length','target_length']}==ref['expected_lengths']
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
    r['saved_vector_provenance']={};score_sources={}
    for key,method in p['quality_schedule']:
        case=cases[key];info=case['input'];source_case=prior['cases'][key]
        assert r['input_freeze_before_model_load'][key]==prior['input_freeze_before_model_load'][key]
        number=p['saved_warm_runs'][key][method];run=prior['runs'][number]
        assert run['number']==number and run['status']=='complete' and run['phase']=='warm'
        assert run['mode']==('batch2' if method=='batch2' else 'single_pair')
        assert run['example_batch_size']==(2 if method=='batch2' else 1)
        record=next(x for x in run['sample_records'] if x['case']==key)
        name=record['vector_key'];full=saved_vectors[name+'_full'];score=saved_vectors[name+'_evaluated']
        assert full.dtype==np.float64 and score.dtype==np.float32 and full.shape==(info['total_length'],) and score.shape==(info['prompt_length'],)
        assert np.array_equal(full[:len(score)].astype(np.float32),score) and np.isfinite(full).all()
        assert sha(full.tobytes())==p['saved_vector_sha256'][key][method]['full']
        assert sha(score.tobytes())==p['saved_vector_sha256'][key][method]['evaluated']
        expected=record['deletion_audit'];order=expected['sorted_keep'];assert sorted(order)==info['keep']
        assert np.all(np.diff(score[order].astype(np.float64))<=0)
        count,extra=divmod(len(order),20);cursor=0;deleted=set();ids=case['ids'].clone()
        receipts=[{'input_sha256':sha(ids[None].numpy().tobytes()),'deleted_positions':[]}]
        groups=[]
        for step in range(20):
            group=order[cursor:cursor+count+(step<extra)];cursor+=len(group);groups.append(group)
            deleted.update(group);ids[group]=tokenizer.eos_token_id
            receipts.append({'input_sha256':sha(ids[None].numpy().tobytes()),'deleted_positions':sorted(deleted)})
        assert receipts==expected['input_receipts'] and groups==expected['groups']
        assert receipts[-1]['deleted_positions']==info['keep']
        dest=key+'_'+method;vectors[dest+'_full']=full.copy();vectors[dest+'_evaluated']=score.copy()
        score_sources[dest]=record
        r['saved_vector_provenance'][dest]={'source_run':number,'source_vector_key':name,'source_mode':run['mode'],
            'source_example_batch_size':run['example_batch_size'],'full_sha256':sha(full.tobytes()),
            'evaluated_sha256':sha(score.tobytes()),'deletion_audit':expected,'source_needle':record['needle'],
            'source_root_logp0_sum_FP32':float(np.sum(record['root_logp0'])),
            'source_root_logp1_sum_FP32':float(np.sum(record['root_logp1'])),
            'scope':'Exact saved warm attribution only; native B1 scorer is newly executed for each actual deletion set.'}
    np.savez_compressed(A/'vectors.npz',**vectors)
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
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'gold':case['gold'],'mapping':case['mapping'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},
            'role':'initialization_only' if key=='niah_mq_q2_0' else 'fixed_original_metric_case'}
    assert p['quality_schedule']==[['niah_mq_q2_1','singleton'],['niah_mq_q2_1','batch2'],
        ['niah_mq_q2_46','batch2'],['niah_mq_q2_46','singleton']]
    for key,method in p['quality_schedule']:
        case=cases[key];info=case['input'];dest=key+'_'+method;score=torch.from_numpy(vectors[dest+'_evaluated'])
        expected=score_sources[dest]['deletion_audit']
        view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        curve={'status':'entered','input_receipts':[],'entered_forwards':0,'returned_forwards':0,
            'source_vector_key':dest,'evaluated_vector_sha256':sha(score.numpy().tobytes()),
            'needle':float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=info['keep'],
                gold_prompt_token_indices=case['gold'],top_fraction=.1)) if case['gold'] else None}
        assert curve['needle']==score_sources[dest]['needle']
        r['cases'][key]['curves'][method]=curve
        def before_score(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scorer_entered']<84
            r['scorer_entered']+=1;curve['entered_forwards']+=1
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(case['ids'])) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=case['ids']).nonzero().flatten().tolist()
            assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
            receipt={'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed}
            curve['input_receipts'].append(receipt);assert receipt==expected['input_receipts'][step]
        def after_score(_module,args,output):
            if output is not None:
                r['scorer_returned']+=1;curve['returned_forwards']+=1
                assert output.logits.dtype==torch.bfloat16
                curve['native_logits_dtype']=str(output.logits.dtype);curve['native_logits_shape']=list(output.logits.shape)
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                loc=frame.f_locals
                for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                    curve[name]=np.asarray(loc[name]).copy().tolist()
                curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
                curve['observed_original_return_metrics']=[float(x) for x in arg]
        handles=[model.register_forward_pre_hook(before_score,with_kwargs=True),model.register_forward_hook(after_score,always_call=True)]
        assert sys.getprofile() is None;r['status']='original_curve_'+dest;save();sys.setprofile(observe)
        try:
            with torch.no_grad():returned=timed('original_curve_'+dest,lambda:faithfulness_test_skip_tokens(view,score[None],
                case['record']['prompt'],case['record']['target'],keep_prompt_token_indices=info['keep'],
                user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[];save()
        curve['return_metrics']=[float(x) for x in returned]
        assert curve['observed_original_return_metrics']==curve['return_metrics']
        assert curve['entered_forwards']==curve['returned_forwards']==21
        assert curve['input_receipts']==expected['input_receipts'] and curve['sorted_keep']==expected['sorted_keep']
        assert all(np.isfinite(curve[name]).all() for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'))
        assert sha(score.numpy().tobytes())==p['saved_vector_sha256'][key][method]['evaluated']
        curve['status']='complete';save()
    assert r['scorer_entered']==r['scorer_returned']==84
    for key in p['quality_cases']:
        row=r['cases'][key];curves=row['curves']
        row['actual_endpoint_scores']={method:[curves[method]['scores'][i] for i in (0,20)] for method in ('singleton','batch2')}
        row['endpoint_batch_curve_minus_singleton_curve']=[row['actual_endpoint_scores']['batch2'][i]-row['actual_endpoint_scores']['singleton'][i] for i in (0,1)]
        row['batch_vector_minus_singleton_vector']={'return_metrics':[a-b for a,b in zip(curves['batch2']['return_metrics'],curves['singleton']['return_metrics'])],
            'needle':curves['batch2']['needle']-curves['singleton']['needle']}
        row['score_target_scope']='Every actual score is original default BF16/FA B1 on the full same fixed original response. Saved attribution native root logs used FP32 logsoftmax and differing endpoint batch shape; they are reported separately and are not scoring substitutes.'
    assert r['DT_entered']==r['DT_returned']==r['finite_FA_calls']==r['finite_FLA_calls']==r['FT_calls']==r['generation_calls']==0
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='true_minibatch_saved_vectors_4original_curves84scores_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['cost_scope']='One model load and unchanged NI0 eager initialization, then four original B1 score curves,84forwards. Zero attribution,finiteFA/FLA,FT,new generation or score cache. Source warm singleton and true batch vectors are exact saved FP64/FP32 arrays with own frozen deletion sets. Score batching is deliberately B1 for both vector sources; this isolates changed attribution/ranking from evaluation batching. All measured initialization/profiling/scoring costs retained.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_entered':r['DT_entered'],'scorer_entered':r['scorer_entered'],
        'scorer_returned':r['scorer_returned'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
