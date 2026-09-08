"""Original author20-step RISE/MAS, fixed5 methods and two actual batched cases."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,io,json,sys,time,traceback,zipfile,inspect,math,concurrent.futures
from pathlib import Path
from collections import Counter
from typing import Any,Dict,Optional,Tuple,List,Sequence
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
r={'status':'running','protocol':p,'model_load_attempts':0,'model_loads':0,'native_B2_forwards':0,'logical_paired_requests':0,
    'cache_hits':0,'original_metric_calls':0,'generation_calls':0,'attribution_calls':0,'calls':[],'methods':{},'requests':[],'artifacts':[]}
start=time.perf_counter();batcher=None;pool=None
def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    return value
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(safe_json(r),indent=2,allow_nan=False));f.replace(HERE/'results.json')
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();tick=time.perf_counter()
    value=fn();torch.cuda.synchronize();r['calls'].append({'kind':label,'seconds':time.perf_counter()-tick,
        'before_bytes':base,'peak_bytes':torch.cuda.max_memory_allocated()});save();return value
def persist(name,values):
    arrays={k:(v.detach().float().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)) for k,v in values.items()}
    f=HERE/(name+'.npz');np.savez_compressed(f,**arrays);r['artifacts'].append({'file':f.name,'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size});save()
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    origin=Path(p['parent_directory']);raw=(origin/'results.json').read_bytes();assert sha(raw)==p['parent_sha256'];previous=json.loads(raw)
    fn=next(n for n in ast.parse((origin/'study.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-guard>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    import numpy as np
    import torch,transformers,flash_attn
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
    import causal_conv1d
    from finite_fla_gpu import verify_native_sources
    from official_fixed_text_inputs import load_author_preparer,prepare_fixed_text
    from author_metric_batching import MetricBatchRequests,BatchedAuthorEvaluator,pack_native_metric_requests,native_batch_response_logprobs
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    author=Path(p['author_root']);trees={}
    for name,digest in p['metric_author_sha256'].items():
        raw=(author/name).read_bytes().replace(b'\r\n',b'\n');assert sha(raw)==digest;trees[name]=ast.parse(raw,filename=str(author/name))
    ns={'__name__':'unchanged_author_metrics','torch':torch,'np':np,'math':math,'Any':Any,'Dict':Dict,'Optional':Optional,'Tuple':Tuple,'List':List,'Sequence':Sequence}
    constants=[x for x in trees['shared_utils.py'].body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['DEFAULT_GENERATE_KWARGS','DEFAULT_PROMPT_TEMPLATE'] for t in x.targets)]
    cls=next(x for x in trees['llm_attr_eval.py'].body if isinstance(x,ast.ClassDef) and x.name=='LLMAttributionEvaluator')
    func=next(x for x in trees['ft_ifr_improve.py'].body if isinstance(x,ast.FunctionDef) and x.name=='faithfulness_test_skip_tokens')
    for nodes in [constants,[cls],[func]]:exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged-author-metrics>','exec'),ns)
    evaluate=ns['faithfulness_test_skip_tokens'];Evaluator=ns['LLMAttributionEvaluator']
    r['model_load_attempts']+=1;save()
    def load():return native.Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',
        device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    model,loading=timed('load_original_Qwen35_BF16',load);r['model_loads']+=1;r['loading']={k:v for k,v in loading.items() if v};assert not r['loading']
    model.eval().requires_grad_(False);assert type(model) is native.Qwen3_5ForConditionalGeneration
    assert model._get_dtype_plan(torch.bfloat16)=={}
    r['parameter_dtypes']=dict(Counter(str(v.dtype) for v in model.parameters()));assert set(r['parameter_dtypes'])=={'torch.bfloat16'}
    layers=model.model.language_model.layers;assert len(layers)==32
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
        else:assert layer.self_attn.config._attn_implementation=='flash_attention_2'
    original_methods={name:type(module).forward for name,module in model.named_modules()}
    def method_audit():
        for name,module in model.named_modules():
            assert 'forward' not in module.__dict__ and type(module).forward is original_methods[name]
            assert not module._forward_hooks and not module._forward_pre_hooks and not module._backward_hooks
    method_audit();r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,'device':torch.cuda.get_device_name()}
    spanfile=Path(p['spans_directory'])/'results.json';assert sha(spanfile.read_bytes())==p['spans_sha256'];spans=json.loads(spanfile.read_bytes())['cases']
    preparer=load_author_preparer(p['author_root'],p['author_source_sha256']);cases=[];evaluators=[]
    for b,row in enumerate(spans):
        tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True,trust_remote_code=False);tokenizer.pad_token=tokenizer.eos_token;assert tokenizer.eos_token_id==248046
        raw=(author/'exp/exp2/data'/f'{row["dataset"]}.jsonl').read_bytes();assert sha(raw)==p['cache_sha256'][row['dataset']]
        record=json.loads(raw.decode().splitlines()[row['index']]);engine=preparer(model,tokenizer);case=prepare_fixed_text(engine,record['prompt'],record['target']);del engine
        assert sha(case['input_ids'].numpy().tobytes())==row['input_metadata']['input_ids_sha256']
        case.update(record=record,user=row['input_metadata']['author_user_positions'],keep=row['mapping']['keep_local_indices']);cases.append(case)
        evaluators.append(Evaluator(model,tokenizer))
    assert [len(c['input_ids']) for c in cases]==[605,368]
    scores={}
    for method,a in p['score_inputs'].items():
        file=Path(a['path']);assert sha(file.read_bytes())==a['sha256'];values=np.load(file,allow_pickle=False)[a['field']]
        scores[method]=[torch.from_numpy(values[b].copy()).float()[case['user']].clamp_min(0)[None,:] for b,case in enumerate(cases)]
    code_names={inspect.unwrap(fn).__code__:label for label,fn in [('FLA',chunk_gated_delta_rule),('FLA_recurrent',fused_recurrent_gated_delta_rule),
        ('FA_varlen',fa.flash_attn_varlen_func),('FA_dense',fa.flash_attn_func),('conv',causal_conv1d.causal_conv1d_fn),
        ('torch_fallback',native.torch_chunk_gated_delta_rule),('torch_recurrent_fallback',native.torch_recurrent_gated_delta_rule)]}
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=2);batcher=MetricBatchRequests();adapters=[BatchedAuthorEvaluator(v,b,batcher) for b,v in enumerate(evaluators)]
    cache={};query_inputs=[];query_masks=[];query_logprobs=[];query_sums=[]
    def worker(b,method):
        trace={}
        def observer(frame,event,value):
            if frame.f_code is evaluate.__code__ and event=='return' and value is not None:
                for k in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
                    trace[k]=frame.f_locals[k].copy()
                trace['sorted_keep']=list(frame.f_locals['sorted_keep']);trace['attr_sum']=float(frame.f_locals['attr_sum'])
        assert sys.getprofile() is None;sys.setprofile(observer)
        try:
            value=evaluate(adapters[b],scores[method][b],cases[b]['record']['prompt'],cases[b]['record']['target'],
                keep_prompt_token_indices=cases[b]['keep'],user_prompt_indices=cases[b]['user'],k=20)
            trace['metrics']=list(map(float,value));return trace
        finally:sys.setprofile(None)
    for method in p['methods']:
        r['methods'][method]={'status':'running','paired_steps':0};futures=[pool.submit(worker,b,method) for b in range(2)];r['original_metric_calls']+=2;save()
        for step in range(21):
            items=batcher.pair();ids,mask,lengths,key=pack_native_metric_requests(items,248046);assert lengths==[605,368]
            for b,item in enumerate(items):
                assert torch.equal(item['response'][0],cases[b]['target_ids'])
                assert item['prompt'].shape[1]==cases[b]['prompt_length']
                original=cases[b]['input_ids'][:cases[b]['prompt_length']];changed=(item['prompt'][0]!=original).nonzero().flatten().tolist()
                allowed={cases[b]['user'][i] for i in cases[b]['keep']}
                assert set(changed)<=allowed and all(int(item['prompt'][0,j])==248046 for j in changed)
            r['logical_paired_requests']+=1
            if key in cache:
                value,index=cache[key];r['cache_hits']+=1;cached=True
            else:
                assert r['native_B2_forwards']<97;r['native_B2_forwards']+=1;counts=Counter()
                def observer(frame,event,_value):
                    if event=='call' and frame.f_code in code_names:counts[code_names[frame.f_code]]+=1
                def forward():
                    assert sys.getprofile() is None;sys.setprofile(observer)
                    try:return native_batch_response_logprobs(model,items,ids,mask)
                    finally:sys.setprofile(None)
                value=timed(f'{method}_step{step}_actual_B2_model_and_native_logsoftmax',forward)
                assert counts=={'FLA':24,'FA_varlen':8,'conv':24},dict(counts)
                index=len(query_inputs);query_inputs.append(ids.numpy());query_masks.append(mask.numpy())
                packed=np.zeros((2,248),dtype=np.float32)
                for b,x in enumerate(value):packed[b,:x.shape[1]]=x.float().cpu().numpy()[0]
                query_logprobs.append(packed);query_sums.append([float(x.sum().cpu()) for x in value]);cache[key]=(value,index);cached=False
                persist('actual_query_'+str(index),{'input_ids':ids,'attention_mask':mask,'target_logprobs':packed,'native_target_sums':query_sums[-1]})
            r['requests'].append({'method':method,'step':step,'input_pair_sha256':key,'query_index':index,'cache_hit':cached,
                'native_target_sums':[float(x.sum().cpu()) for x in value]})
            # Only actual native outputs (possibly an exact earlier paired query)
            # are delivered to unchanged author metric code, still BF16/GPU.
            for item,x in zip(items,value):item['promise'].set_result(x)
            r['methods'][method]['paired_steps']=step+1;save()
        result=[f.result(timeout=30) for f in futures]
        for b,row in enumerate(result):
            row['equal_original_and_final_baseline']=bool(row['scores'][0]==row['scores'][-1])
            persist(method+f'_sample{b}_author_curves',{k:v for k,v in row.items() if k!='equal_original_and_final_baseline'})
        r['methods'][method].update(status='complete',samples=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items()} for x in result]);save()
    assert r['original_metric_calls']==10 and r['logical_paired_requests']==105 and r['native_B2_forwards']<=97
    r['all_unique_query_count']=len(cache);r['cache_sequence_hits']=2*r['cache_hits'];method_audit()
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    r['status']='same_model_original_RISE_MAS_five_methods_two_cases_executed'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    if batcher is not None:batcher.abort()
    if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    r['job_seconds_before_bundle']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[x['file'] for x in r['artifacts']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'error':r.get('error')}),flush=True)
