"""Bounded nested CUDA-event timing around existing native DT components.

No replacement arithmetic, synchronization inside an operator, or extra
model/DT/optimizer calls. Outer original stage times include these components.
"""
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-dt-cost-breakdown-20261016-v1'


def install_nested(worker):
    import collections,hashlib,inspect,json,os,sys,time
    from pathlib import Path
    import psutil,torch
    import reward_readout as owner
    from torch._dynamo import utils
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    installed=json.loads((AUDIT/'observation-installed.json').read_bytes())
    identities={v['pid']:v['birth'] for v in installed['workers']}
    assert psutil.Process().create_time()==identities[os.getpid()]
    cls=owner.DirectActionTargetReadout
    outer=cls.trajectories
    original=inspect.getclosurevars(outer).nonlocals['original']
    assert hasattr(cls,'_passive_DT_cost_v1')
    def observed(instance,*args,**kwargs):
        runner=instance.runner;module=sys.modules[type(runner).__module__]
        targets=[(runner.boundaries,'mlp','finite_MLP')]
        for name,label in [('gdn_finite_pullback','finite_GDN_mixer'),('attention_finite_pullback','finite_FA_mixer')]:
            assert hasattr(module,name);targets.append((module,name,label))
        restored=[];events=[]
        def wrap(function,label):
            def measured(*aa,**kk):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                counts={k:len(v) for k,v in utils.compilation_time_metrics.items()}
                tick=time.perf_counter();start.record()
                try:return function(*aa,**kk)
                finally:
                    end.record();events.append(dict(label=label,start=start,end=end,
                        Python_launch_and_compile_seconds=time.perf_counter()-tick,
                        compiler_inclusive_seconds={k:list(v[counts.get(k,0):]) for k,v in utils.compilation_time_metrics.items() if len(v)>counts.get(k,0)}))
            return measured
        for obj,name,label in targets:
            function=getattr(obj,name);restored.append((obj,name,function));setattr(obj,name,wrap(function,label))
        tick=time.perf_counter()
        record=dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),started_unix=time.time(),
            additional_model_DT_backward_optimizer_calls=0,nested_within_original_decoder_stage=True,
            added_timing_only_CUDA_events=0,events=[],status='observing')
        try:
            result=outer(instance,*args,**kwargs)
            # The native attribute owner has already completed its event
            # accounting and diagnostics. No synchronization is inserted into
            # a component's execution or between its tensor operations.
            for event in events:
                record['events'].append(dict(label=event['label'],stream_elapsed_seconds=event['start'].elapsed_time(event['end'])/1000,
                    Python_launch_and_compile_seconds=event['Python_launch_and_compile_seconds'],
                    compiler_inclusive_seconds=event['compiler_inclusive_seconds']))
            record.update(status='original_DT_nested_timings_completed',wall_seconds=time.perf_counter()-tick,
                          added_timing_only_CUDA_events=2*len(events))
            return result
        except BaseException as exc:
            record.update(status='native_or_observation_raised',error=repr(exc));raise
        finally:
            for obj,name,function in restored:setattr(obj,name,function)
            record.update(completed_unix=time.time(),original_components_restored=all(getattr(obj,name) is function for obj,name,function in restored),
                original_trajectories_restored=cls.trajectories is original)
            (AUDIT/f'rank{w.rank}-nested-components.json').write_text(json.dumps(record,indent=2)+'\n')
    cls.trajectories=observed
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,status='nested_native_component_timing_installed')


if __name__=='__main__':
    import psutil,ray
    job=json.loads((AUDIT/'job.json').read_bytes());p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
    assert (AUDIT/'observation-installed.json').exists()
    gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2
        values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install_nested) for a in names],timeout=1200)
        (AUDIT/'nested-observation-installed.json').write_text(json.dumps(dict(unix=time.time(),workers=values),indent=2)+'\n')
    finally:ray.shutdown()
