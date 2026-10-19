"""Publish existing finite-runner metadata for one future native readout.

This uses the original worker metadata RPC. It adds no DT/model/optimizer
call, CUDA event, synchronization, memory clear, numerical option or RNG use.
The existing None observer argument and all return objects remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
wrapper=Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()=='986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code=wrapper.read_text()
anchor="exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor)==1
observation='''    producer=getattr(actor,'_deltatrace_producer',None)
    readout=getattr(producer,'direct_readout',None)
    runner=getattr(producer,'runner',None)
    installed=False
    if readout is not None and runner is not None and not getattr(readout,'_existing_DT_timing_scope_installed',False):
        import json,time
        native_readout=readout.trajectories
        native_attribute=runner.attribute
        def observe_one_existing_readout(*args,**kwargs):
            count=[0]
            started=time.time()
            def publish_existing_attribute(*native_args,**native_kwargs):
                began=time.time()
                returned=native_attribute(*native_args,**native_kwargs)
                ended=time.time()
                try:
                    detail=returned[1]
                    paired=native_args[0] if native_args else native_kwargs['paired_ids']
                    report=dict(pid=os.getpid(),rank=actor.rank,minibatch=count[0]+1,
                        started=began,finished=ended,paired_shape=list(paired.shape),
                        complete_attribution_seconds=detail.get('complete_attribution_seconds_with_diagnostics'),
                        peak_allocated=detail.get('peak_allocated'),peak_reserved=detail.get('peak_reserved'),
                        calls=detail.get('calls'),controller_diagnostic_scheduling=detail.get('controller_diagnostic_scheduling'),
                        original_observer_is_none=native_kwargs.get('observer') is None,
                        additional_DT_calls=0,new_CUDA_events=0,original_return_objects_unchanged=True,
                        timing_scope='Original stage host enqueue/stream elapsed; stream elapsed is not active GPU kernel time; nested ranges must not be summed as driver wall time')
                    print('[native_DT_existing_timing] '+json.dumps(report),flush=True)
                except Exception as error:
                    print('[native_DT_existing_timing_observer_error] '+repr(error),flush=True)
                count[0]+=1
                return returned
            runner.attribute=publish_existing_attribute
            try:
                return native_readout(*args,**kwargs)
            finally:
                runner.attribute=native_attribute
                readout.trajectories=native_readout
                try:
                    print('[native_DT_existing_timing_scope] '+json.dumps(dict(pid=os.getpid(),rank=actor.rank,
                        started=started,finished=time.time(),native_minibatches=count[0],
                        restored_original_methods=True,additional_DT_calls=0)),flush=True)
                except Exception:
                    pass
        readout.trajectories=observe_one_existing_readout
        readout._existing_DT_timing_scope_installed=True
        installed=True
    result['existing_DT_timing_observer_installed']=installed
    result['existing_DT_timing_observer_scope']='one future existing readout per rank, then restore original methods'
    result['existing_DT_timing_observer_additional_calls']=0
    return result'''
code=code.replace(anchor,"assert code.count('    return result')==1\ncode=code.replace('    return result',"+repr(observation)+")\n"+anchor)
exec(compile(code,str(wrapper),'exec'),globals())
# Keep a separate explicit runtime-override receipt for this passive observer;
# the original binding record remains the original metadata component output.
observed=globals().get('results',[])
if observed:
    record=dict(status='completed_existing_DT_metadata_observer_installation',collected=time.time(),
        source=str(Path(__file__)),source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_metadata_receipt=str(globals().get('receipt')),
        workers=[{key:r.get(key) for key in ['pid','birth','rank','existing_DT_timing_observer_installed']} for r in observed],
        additional_model_calls=0,additional_DT_calls=0,optimizer_updates=0,new_CUDA_events=0,
        computational_source_changes=0,scope='Publish already-computed native metadata for one future original readout per rank; automatic restoration')
    (AUDIT/'existing-DT-metadata-observer-completed.json').write_text(json.dumps(record,indent=2)+'\n')
