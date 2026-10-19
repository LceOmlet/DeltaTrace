"""Read the loaded vLLM config through the existing worker metadata RPC."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                'loaded-generation-config-readonly-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import inspect,torch
    engine=getattr(getattr(actor,'rollout',None),'inference_engine',None)
    llm=getattr(engine,'llm_engine',None)
    core=getattr(getattr(llm,'engine_core',None),'engine_core',None)
    executor=getattr(core,'model_executor',None)
    worker=getattr(getattr(executor,'driver_worker',None),'worker',None)
    runner=getattr(worker,'model_runner',None)
    config=getattr(runner,'vllm_config',None)
    fields={
        'model_config':('model','dtype','max_model_len','enforce_eager'),
        'scheduler_config':('async_scheduling','max_num_seqs','max_num_batched_tokens','enable_chunked_prefill'),
        'lora_config':('max_lora_rank','max_loras','max_cpu_loras','lora_dtype','fully_sharded_loras'),
        'compilation_config':('cudagraph_mode','cudagraph_capture_sizes'),
        'parallel_config':('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')}
    values={}
    for section,names in fields.items():
        current=getattr(config,section,None)
        values[section]={name:(value if isinstance(value,(int,float,bool,str,list,tuple)) or value is None else str(value))
            for name in names for value in (getattr(current,name,None),)}
    scheduler=getattr(core,'scheduler',None)
    cls=type(scheduler) if scheduler is not None else None
    source=Path(inspect.getfile(type(runner))) if runner is not None else None
    result['loaded_generation_config']=dict(
        runner_found=runner is not None,values=values,
        actual_scheduler_class=(cls.__module__+'.'+cls.__qualname__) if cls is not None else None,
        device_name=torch.cuda.get_device_name(),
        tuned_config_folder=os.environ.get('VLLM_TUNED_CONFIG_FOLDER'),
        runner_source=str(source) if source else None,
        runner_source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        model_calls=0,DT_calls=0,optimizer_updates=0,resource_mutations=0,
        scope='Read actual loaded configs only. Missing fields remain null; no defaults inferred.')
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"
                    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
