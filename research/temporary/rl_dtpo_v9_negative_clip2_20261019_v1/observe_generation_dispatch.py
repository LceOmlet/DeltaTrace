"""Read 512 existing native vLLM dispatch decisions per rank, then restore.

The original dispatcher returns its original objects. No model call, graph
capture, tensor operation, synchronization, scheduling option or RNG changes.
This distinguishes actual graph use from configured graph support.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                 'generation-dispatch-passive-v4')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import collections,inspect,json,time
    engine=getattr(getattr(actor,'rollout',None),'inference_engine',None)
    llm=getattr(engine,'llm_engine',None)
    client=getattr(llm,'engine_core',None)
    core=getattr(client,'engine_core',None)
    executor=getattr(core,'model_executor',None)
    wrapper=getattr(executor,'driver_worker',None)
    worker=getattr(wrapper,'worker',None)
    runner=getattr(worker,'model_runner',None)
    result['generation_dispatch_observer_installed']=False
    if runner is not None and not getattr(runner,'_native_dispatch_observer_v4',False):
        original=runner._determine_batch_execution_and_padding
        signature=inspect.signature(original)
        counts=collections.Counter()
        state={'calls':0,'started':None}
        def observe_existing_dispatch(*args,**kwargs):
            returned=original(*args,**kwargs)
            try:
                if state['started'] is None:state['started']=time.time()
                parameters=kwargs if not args else signature.bind(*args,**kwargs).arguments
                mode,descriptor=returned[:2]
                key=(str(mode),int(parameters['num_tokens']),int(parameters['num_reqs']),
                    int(parameters['max_num_scheduled_tokens']),int(descriptor.num_tokens))
                counts[key]+=1
                state['calls']+=1
                if state['calls']>=512:
                    runner._determine_batch_execution_and_padding=original
                    print('[native_generation_dispatch] '+json.dumps(dict(
                        pid=os.getpid(),rank=actor.rank,started=state['started'],finished=time.time(),
                        existing_dispatch_calls=state['calls'],additional_model_calls=0,
                        scheduling_configuration_changes=0,original_return_objects_unchanged=True,
                        restored_original_method=True,counts=[dict(runtime_mode=k[0],
                            unpadded_tokens=k[1],requests=k[2],max_scheduled_per_request=k[3],
                            padded_tokens=k[4],calls=v) for k,v in sorted(counts.items())])),flush=True)
            except Exception as error:
                runner._determine_batch_execution_and_padding=original
                print('[native_generation_dispatch_observer_error] '+repr(error),flush=True)
            return returned
        runner._determine_batch_execution_and_padding=observe_existing_dispatch
        runner._native_dispatch_observer_v4=True
        result['generation_dispatch_observer_installed']=True
        result['generation_dispatch_native_source']=inspect.getfile(type(runner))
        result['generation_dispatch_native_source_SHA256']=hashlib.sha256(
            Path(result['generation_dispatch_native_source']).read_bytes()).hexdigest()
        model=runner.model
        entries=getattr(model,'concrete_cudagraph_entries',None)
        result['generation_existing_graph_entries']=len(entries) if entries is not None else None
    result['generation_dispatch_existing_call_limit']=512
    result['additional_generation_calls']=0
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\n"
    "code=code.replace('    return result'," + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
