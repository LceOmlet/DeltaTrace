"""Complete-call paired DT/FT cost on the DT paper's 16 development examples."""
import argparse,copy,gc,hashlib,inspect,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids_sha=lambda x:hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()

def save(path,value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_bytes())
    plan=protocol['efficiency'];code=a.run_root/'repo_dynamic'
    environment_path=a.run_root/'environment_dynamic.json'
    env=json.loads(environment_path.read_bytes())['qwen35']
    full=a.run_root/'full_dynamic_with_ifr'
    identity=json.loads((full/'identity.json').read_bytes())
    assert sha(environment_path)==identity['environment_sha256']
    assert sha(code/'deltatrace/clean/sources.json')==identity['clean_sources_sha256']
    for name,spec in json.loads((code/'deltatrace/clean/sources.json').read_bytes())['models']['qwen35']['files'].items():
        assert sha(code/name)==spec['sha256'],name
    for name,digest in env['official_extension_blob_sha1'].items():
        b=(Path(env['ft_extension_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==digest,name
    for name,digest in env['normalization_owner_sha256'].items():
        assert sha(name)==digest,name
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0',
        TRITON_CACHE_DIR=env['triton_cache'],TORCHINDUCTOR_CACHE_DIR=env['inductor_cache'])
    os.environ.update(env['runtime_environment'])
    os.environ['PATH']=str(a.run_root/'env/bin')+':/opt/conda/bin:/usr/bin:/bin:'+os.environ.get('PATH','')
    sys.path.insert(0,env['official_root'])
    sys.path.insert(0,str(code/'deltatrace/clean/qwen35'))
    sys.path.append(env['ft_extension_root'])
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from exp.exp2 import dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from flashtrace import FlashTrace
    from qwen35_clean_runner import make_qwen35_clean_runner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    reuse=json.loads((full/'process_reuse.json').read_bytes())
    for name in ('cache_size_limit','accumulated_cache_size_limit'):
        setattr(torch._dynamo.config,name,reuse[name])
    verify_native_sources(env['native_stage_source_sha256'])
    examples=[];references={};reference_vectors={}
    for task in dict(plan['selection']).keys():
        raw=json.loads((full/task/'results.json').read_bytes())
        assert raw['status']=='complete'
        assert raw['driver_sha256']==identity['driver_sha256']
        assert sha(full/task/'vectors.npz')==raw['vectors_sha256']
        rows={r['index']:r for r in raw['cases']}
        cache=dataset_utils.load_cached(Path(env['official_root'])/'exp/exp2/data'/f'{task}.jsonl')
        with np.load(full/task/'vectors.npz',allow_pickle=False) as vectors:
            for selected_task,index in plan['selection']:
                if selected_task!=task:continue
                key=f'{task}_{index}';references[key]=rows[index]
                for method,suffix in [('DT','DT_signed_full'),('FT_K1','FT_K1_prompt'),('FT_K3','FT_K3_prompt')]:
                    # MoreHopQA lacks gold, so full quality runs have only FT K1.
                    if key+'_'+suffix in vectors:
                        reference_vectors[key+'_'+method]=vectors[key+'_'+suffix].copy()
                examples.append((task,index,copy.deepcopy(cache[index])))
    assert [(t,i) for t,i,_ in examples]==[tuple(x) for x in plan['selection']]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_bytes(a.protocol.read_bytes())
    (a.output/'benchmark_complete_calls.py').write_bytes(Path(__file__).read_bytes())
    report=dict(status='loading',protocol_sha256=sha(a.protocol),driver_sha256=sha(__file__),
        environment_sha256=sha(environment_path),clean_sources_sha256=identity['clean_sources_sha256'],
        generation_calls=0,examples=[],separate_costs=[],full_run_identity=identity,
        runtime=dict(hostname=os.uname().nodename,pid=os.getpid(),torch=torch.__version__,
            transformers=sys.modules['transformers'].__version__,dtype='bfloat16'),gpu_ownership=[])
    saved={}
    def persist():
        save(a.output/'results.json',report)
    def exclusive_gpu(label):
        state=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True)
        pids=sorted(set(map(int,re.findall(r'^\|\s+0\s+(\d+)\s+\S+',state.stdout,re.MULTILINE))))
        report['gpu_ownership'].append(dict(label=label,pids=pids,expected_pid=os.getpid(),snapshot=state.stdout))
        assert pids==[os.getpid()],f'GPU cost measurement requires an isolated device; got {pids}'
    persist()
    try:
        start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token
        model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
        torch.cuda.synchronize()
        model.eval().requires_grad_(False)
        exclusive_gpu('after_model_load')
        assert sha(inspect.getfile(type(model)))==env['native_model_sha256']
        report['separate_costs'].append(dict(name='model_load',seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        originals={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        metadata_start=time.perf_counter()
        for ordinal,(task,index,ex) in enumerate(examples):
            ex.indices_to_explain=ex.sink_span=ex.thinking_span=None
            ex=dataset_utils.attach_spans_from_answer(ex,tokenizer)
            ex.indices_to_explain=list(ex.sink_span)
            examples[ordinal]=(task,index,ex)
        report['separate_costs'].append(dict(name='fixed_dataset_target_metadata',seconds=time.perf_counter()-metadata_start))
        init=torch.tensor([references[f'{examples[0][0]}_{examples[0][1]}']['input_ids']],device=model.device)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            initial=model(input_ids=init,attention_mask=torch.ones_like(init),use_cache=False)
        torch.cuda.synchronize()
        report['separate_costs'].append(dict(name='original_eager_initialization',seconds=time.perf_counter()-start))
        del initial,init
        method_setup_start=time.perf_counter()
        finite=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        options=env.get('dt_compiler_options')
        dt=make_qwen35_clean_runner(model,finite,
            make_compiled_finite_pullback(reuse_scalar_products=False,dynamic_shapes=True,compiler_options=options),
            dynamic_shapes=True,compiler_options=options)
        for field in ('norm_gate_rules','finite_fla_by_layer','attention_pv_rules','key_norm_by_layer'):
            assert getattr(dt,field)=={}
        tracer=FlashTrace(model,tokenizer,use_chat_template=False)
        formatter=LLMAttributionEvaluator(model,tokenizer)
        torch.cuda.synchronize()
        report['separate_costs'].append(dict(name='method_initialization',seconds=time.perf_counter()-method_setup_start))
        for ordinal,(task,index,ex) in enumerate(examples):
            key=f'{task}_{index}';expected=references[key]
            exclusive_gpu(key+'_before')
            case=dict(dataset=task,index=index,input_sha256=expected['input_sha256'],calls=[])
            report['examples'].append(case)
            for repeat in range(4):
                offset=(ordinal+repeat)%len(plan['methods'])
                order=plan['methods'][offset:]+plan['methods'][:offset]
                for method in order:
                    model.set_attn_implementation('flash_attention_2' if method=='DT' else 'eager')
                    for name,module in model.named_modules():
                        assert (type(module).forward,module.forward)==originals[name]
                    gc.collect();torch.cuda.empty_cache()
                    observed=[]
                    def hook(_model,args,kw):
                        value=kw['input_ids'].detach().cpu().numpy()
                        observed.append(dict(shape=list(value.shape),sha256=ids_sha(value)))
                    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    graphs=torch._dynamo.utils.counters['stats']['unique_graphs']
                    start=time.perf_counter()
                    try:
                        if method=='DT':
                            engine=ft.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
                            ids,mask,prompt_len,gen_len=engine._ensure_generation(ex.prompt,ex.target)
                            positions=list(engine.user_prompt_indices)
                            keep=ft.keep_token_indices(engine.user_prompt_tokens)
                            assert ids_sha(ids.detach().cpu().numpy())==expected['input_sha256']
                            assert positions==expected['user_positions'] and keep==expected['keep']
                            base=ids.clone();base[0,[positions[j] for j in keep]]=tokenizer.eos_token_id
                            pair=torch.cat((base,ids))
                            selection=PackedAnswerTargets([dict(target_ids=ids[0,prompt_len:].cpu(),prompt_length=prompt_len)],
                                [list(range(gen_len))],ids.shape[1],model.device)
                            prep_peak=torch.cuda.max_memory_allocated()
                            value,detail=dt.attribute(pair,torch.ones_like(pair),selection,select_output_rows=True,observer=None)
                            returned=value[0].detach().cpu().numpy().copy()
                            del value,detail,selection,pair,base,engine,ids,mask
                        else:
                            # The public FT call performs its own tokenization. Do
                            # not charge it a second external tokenization pass.
                            formatted=formatter.format_prompt(' '+ex.prompt)
                            positions=expected['user_positions'];keep=expected['keep']
                            prep_peak=torch.cuda.max_memory_allocated()
                            hops=int(method[-1])
                            value=tracer.trace(prompt=formatted,target=ex.target,output_span=tuple(ex.sink_span),
                                reasoning_span=tuple(ex.thinking_span),hops=hops,method='flashtrace')
                            returned=np.asarray(value.scores,dtype=np.float32)[positions].copy()
                            del value
                        torch.cuda.synchronize()
                        seconds=time.perf_counter()-start
                        peak=max(prep_peak,torch.cuda.max_memory_allocated())
                        new_graphs=torch._dynamo.utils.counters['stats']['unique_graphs']-graphs
                    finally:
                        handle.remove()
                    assert len(observed)==1 and np.isfinite(returned).all()
                    original=np.asarray(expected['input_ids'],dtype=np.int64)
                    if method=='DT':
                        reference=original.copy();reference[np.asarray(positions)[keep]]=tokenizer.eos_token_id
                        wanted=np.stack((reference,original))
                    else:wanted=original[None]
                    assert observed==[dict(shape=list(wanted.shape),sha256=ids_sha(wanted))]
                    name=f'{key}_{method}_r{repeat}'
                    saved[name]=returned
                    reference_value=reference_vectors.get(key+'_'+method)
                    comparison=None
                    if reference_value is not None:
                        difference=returned.astype(np.float64)-reference_value.astype(np.float64)
                        comparison=dict(bitwise_equal=bool(np.array_equal(returned,reference_value)),
                            relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(reference_value),1e-30)),
                            max_absolute_error=float(np.max(np.abs(difference))))
                    call=dict(method=method,repeat=repeat,warmup=repeat==0,seconds=seconds,
                        peak_allocated_bytes=peak,new_torch_graphs=new_graphs,actual_model_inputs=observed,
                        vector_key=name,full_quality_vector_comparison=comparison)
                    case['calls'].append(call)
                    report['status']=f'{key}_{method}_r{repeat}'
                    np.savez_compressed(a.output/'vectors.npz',**saved)
                    persist()
                    print(json.dumps(dict(case=key,method=method,repeat=repeat,seconds=seconds)),flush=True)
                    del returned
            case['status']='complete'
            exclusive_gpu(key+'_after')
        report['status']='complete'
        report['vectors_sha256']=sha(a.output/'vectors.npz')
        assert len(report['examples'])==16 and sum(len(c['calls']) for c in report['examples'])==192
        persist()
    except Exception:
        report.update(status='failed',error=traceback.format_exc());persist();raise
if __name__=='__main__':main()
