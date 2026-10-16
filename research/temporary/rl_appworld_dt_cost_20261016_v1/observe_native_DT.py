"""Passive original DT stage and compiler accounting in an isolated native trial.

No extra model, tensor, attribution, backward, optimizer or collective calls.
The original method's exact return object and the original formulas are kept.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-dt-cost-breakdown-20261016-v1'


def install(worker):
    import inspect
    from types import MethodType
    import psutil
    import reward_readout as owner
    from torch._dynamo import utils
    actual=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    assert os.environ['DT_TASK']=='AppWorld'
    assert actual.config.actor.ppo_micro_batch_size_per_gpu==4
    assert (actual.config.model.lora_rank,actual.config.model.lora_alpha)==(8,16)
    original=owner.DirectActionTargetReadout.trajectories
    assert not hasattr(owner.DirectActionTargetReadout,'_passive_DT_cost_v1')
    owner.DirectActionTargetReadout._passive_DT_cost_v1=True
    binding=dict(path=owner.__file__,SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
                 method_source=inspect.getsourcefile(original))
    def compiler_counts():return {k:len(v) for k,v in utils.compilation_time_metrics.items()}
    def compiler_delta(before):
        return {k:list(v[before.get(k,0):]) for k,v in utils.compilation_time_metrics.items()
                if len(v)>before.get(k,0)}
    def observed(instance,*args,**kwargs):
        runner=instance.runner
        assert runner.defer_diagnostics
        base=runner.attribute
        existed='attribute' in vars(runner);previous=vars(runner).get('attribute')
        record=dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=actual.rank,
            started_unix=time.time(),scheduler_before=actual.actor_lr_scheduler.last_epoch,
            source=binding,batches=[],additional_model_DT_tensor_optimizer_collective_calls=0,
            original_formulas_and_return_unchanged=True)
        record['runner_flags']={k:getattr(runner,k,None) for k in ('defer_diagnostics','copy_replay_captures',
            'offload_replay_mixer','pin_replay_host','pin_root_host','reuse_native_prefix',
            'fa_coefficient_suffix','gdn_coefficient_suffix','compact_gdn_captures')}
        record['DT_module_bindings']={name:dict(path=m.__file__,SHA256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
            for name in ('qwen35_dense_finite_runner','qwen35_decoder_finite','qwen35_gdn_finite')
            if (m:=sys.modules.get(name)) is not None}
        before=compiler_counts()
        def attribute(this,*aa,**kk):
            counts=compiler_counts();tick=time.perf_counter()
            result=base(*aa,**kk)
            record['batches'].append(dict(input_shape=list(aa[0].shape if aa else kk['paired_ids'].shape),
                original_attribute_wall_seconds=time.perf_counter()-tick,
                compiler_inclusive_durations_seconds=compiler_delta(counts),owner_info=result[1]))
            # Persist between native batches without touching any result tensor.
            (AUDIT/f'rank{actual.rank}-DT-partial.json').write_text(json.dumps(record,indent=2)+'\n')
            return result
        runner.attribute=MethodType(attribute,runner)
        tick=time.perf_counter()
        try:
            result=original(instance,*args,**kwargs)
            record.update(original_trajectories_wall_seconds=time.perf_counter()-tick,
                original_readout_report=instance.last_report,status='original_native_DT_returned',
                compiler_inclusive_durations_seconds=compiler_delta(before))
            return result
        except BaseException as exc:
            record.update(status='original_native_DT_raised',error=repr(exc));raise
        finally:
            if existed:runner.attribute=previous
            else:del runner.attribute
            owner.DirectActionTargetReadout.trajectories=original
            record.update(completed_unix=time.time(),scheduler_after=actual.actor_lr_scheduler.last_epoch,
                original_attribute_restored=runner.attribute==base,
                original_trajectories_restored=owner.DirectActionTargetReadout.trajectories is original)
            (AUDIT/f'rank{actual.rank}-DT-cost.json').write_text(json.dumps(record,indent=2)+'\n')
    owner.DirectActionTargetReadout.trajectories=observed
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=actual.rank,
                original_binding=binding,status='passive_native_DT_accounting_installed')


if __name__=='__main__':
    import psutil
    import ray
    job=json.loads((AUDIT/'job.json').read_bytes())
    p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
    tick=time.monotonic();gcs=None
    while time.monotonic()-tick<300:
        gcs=next((c for c in p.children(recursive=True) if c.name()=='gcs_server'),None)
        if gcs is not None:break
        time.sleep(1)
    assert gcs is not None
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        tick=time.monotonic();names=[]
        while time.monotonic()-tick<300:
            names=[a for a in ray.util.list_named_actors(all_namespaces=True)
                   if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            if len(names)==2:break
            time.sleep(1)
        assert len(names)==2
        values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                       for a in names],timeout=1200)
        assert {v['rank'] for v in values}=={0,1}
        (AUDIT/'observation-installed.json').write_text(json.dumps(dict(unix=time.time(),workers=values),indent=2)+'\n')
    finally:ray.shutdown()
