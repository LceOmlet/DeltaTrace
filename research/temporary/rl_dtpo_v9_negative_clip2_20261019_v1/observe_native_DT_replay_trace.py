"""Profile two native replay layers in one naturally scheduled long B4.

The owner keeps all original forward/finite calls and return objects. The
profile covers FA31, finite31, and GDN30; no attribution call is repeated.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'native-DT-replay-trace-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import json,time,torch
    producer=getattr(actor,'_deltatrace_producer',None)
    runner=getattr(producer,'runner',None)
    result['native_DT_replay_trace_installed']=False
    if actor.rank==0 and runner is not None and not getattr(runner,'_native_replay_trace_v1',False):
        readout=producer.direct_readout
        native_readout=readout.trajectories
        native_attribute=runner.attribute
        model=runner.model
        native_replay=model.replay_finite_layer
        layers=model.model.language_model.layers
        output=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/native-DT-replay-trace-rank0.json')
        assert not output.exists()
        def observe_one_readout(*args,**kwargs):
            state=dict(selected=False,active=False,completed=False,calls=0,errors=[])
            profile=None
            def observe_attribute(*native_args,**native_kwargs):
                nonlocal profile
                state['calls']+=1
                paired=native_args[0] if native_args else native_kwargs['paired_ids']
                if not state['selected'] and paired.shape[1]>=6000:
                    state['selected']=True
                    state['paired_shape']=list(paired.shape)
                    def observe_replay(layer,replay):
                        nonlocal profile
                        if layer is layers[31]:
                            try:
                                profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False,with_flops=False)
                                profile.start();state['active']=True;state['started']=time.time()
                            except Exception as error:state['errors'].append(repr(error))
                        try:
                            return native_replay(layer,replay)
                        finally:
                            if layer is layers[30] and state['active']:
                                state['active']=False
                                try:
                                    profile.stop();state['finished']=time.time()
                                    profile.export_chrome_trace(str(output));state['completed']=True
                                except Exception as error:state['errors'].append(repr(error))
                    model.replay_finite_layer=observe_replay
                try:
                    return native_attribute(*native_args,**native_kwargs)
                finally:
                    model.replay_finite_layer=native_replay
            runner.attribute=observe_attribute
            try:
                return native_readout(*args,**kwargs)
            finally:
                model.replay_finite_layer=native_replay
                runner.attribute=native_attribute;readout.trajectories=native_readout
                if state['active']:
                    state['active']=False
                    try:profile.stop()
                    except Exception as error:state['errors'].append(repr(error))
                record=dict(status='completed_native_replay_trace' if state['completed'] else 'native_replay_trace_not_completed',pid=os.getpid(),rank=actor.rank,state=state,trace=str(output),restored_original_methods=True,additional_model_calls=0,additional_DT_calls=0,optimizer_calls=0,exclude_from_uninstrumented_performance_comparison=True,scope='Native FA31 replay plus finite31 and GDN30 replay in first natural B4 with paired extent>=6000; original Torch profiler, no new forward or finite call')
                output.with_name('native-DT-replay-trace-receipt.json').write_text(json.dumps(record,indent=2)+chr(10))
                print('[native_DT_replay_trace] '+json.dumps(record),flush=True)
        readout.trajectories=observe_one_readout
        runner._native_replay_trace_v1=True
        result['native_DT_replay_trace_installed']=True
    result['additional_DT_calls']=0
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result'," + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
