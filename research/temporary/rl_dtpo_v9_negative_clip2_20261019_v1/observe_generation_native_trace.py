"""Profile 64 naturally scheduled native generation steps on rank0 only.

Use the installed torch.profiler API. No new model/environment/optimizer call,
sampling change or graph capture. Exclude this instrumented interval from
production timing claims; the original dispatcher objects are returned.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                 'generation-native-trace-v4')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import json,time,torch
    engine=getattr(getattr(actor,'rollout',None),'inference_engine',None)
    llm=getattr(engine,'llm_engine',None)
    core=getattr(getattr(llm,'engine_core',None),'engine_core',None)
    executor=getattr(core,'model_executor',None)
    worker=getattr(getattr(executor,'driver_worker',None),'worker',None)
    runner=getattr(worker,'model_runner',None)
    result['generation_native_trace_installed']=False
    result['generation_native_trace_rank']=0
    if actor.rank==0 and runner is not None and not getattr(runner,'_native_generation_trace_v4',False):
        activities={torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA}
        assert activities.issubset(torch.profiler.supported_activities())
        native=runner._determine_batch_execution_and_padding
        profile=torch.profiler.profile(activities=list(activities),record_shapes=False,
            profile_memory=False,with_stack=False,with_flops=False)
        state={'calls':0,'started':None,'active':False}
        output=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/native-generation-64step-trace-rank0.json')
        assert not output.exists()
        def observe_native_dispatch(*args,**kwargs):
            returned=native(*args,**kwargs)
            state['calls']+=1
            try:
                if state['calls']==64:
                    state['started']=time.time()
                    profile.start()
                    state['active']=True
                elif state['calls']==128:
                    runner._determine_batch_execution_and_padding=native
                    profile.stop()
                    state['active']=False
                    finished=time.time()
                    profile.export_chrome_trace(str(output))
                    receipt=dict(status='completed_bounded_native_generation_trace',
                        pid=os.getpid(),rank=actor.rank,started=state['started'],finished=finished,
                        profiled_existing_dispatch_steps=64,trace=str(output),
                        additional_model_calls=0,additional_generation_calls=0,
                        original_return_objects_unchanged=True,restored_original_method=True,
                        exclude_from_production_speed_measurement=True,
                        scope='Original torch.profiler CPU/CUDA activity; no shape/memory/stack/FLOPS collection')
                    output.with_name('native-generation-64step-trace-receipt.json').write_text(json.dumps(receipt,indent=2)+chr(10))
                    print('[native_generation_trace] '+json.dumps(receipt),flush=True)
            except Exception as error:
                runner._determine_batch_execution_and_padding=native
                if state['active']:
                    try:profile.stop()
                    except Exception:pass
                print('[native_generation_trace_observer_error] '+repr(error),flush=True)
            return returned
        runner._determine_batch_execution_and_padding=observe_native_dispatch
        runner._native_generation_trace_v4=True
        result['generation_native_trace_installed']=True
    result['additional_generation_calls']=0
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\n"
    "code=code.replace('    return result'," + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
