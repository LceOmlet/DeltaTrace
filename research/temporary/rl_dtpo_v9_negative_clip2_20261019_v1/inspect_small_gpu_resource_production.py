"""Read counters from the installed original-owner resource selector.

Reuse the existing worker metadata RPC; do not install another observer,
invoke DT, or change storage flags.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                'small-GPU-resident-production-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    producer=getattr(actor,'_deltatrace_producer',None)
    assert producer is not None
    runner=producer.runner
    state=getattr(runner,'_bounded_native_mixer_storage',None)
    assert state is not None
    result['small_GPU_resident_production']=dict(
        counters={k:v for k,v in state.items() if k!='original_attribute'},
        current_flags=dict(pin_root_host=runner.pin_root_host,
            pin_replay_host=runner.pin_replay_host,offload_replay_mixer=runner.offload_replay_mixer),
        installed_attribute_name=runner.attribute.__name__,
        actor_scheduler_epoch=actor.actor_lr_scheduler.last_epoch,
        deployment_receipt='runtime-overrides/small-GPU-resident-resource-override-completed.json',
        additional_DT_calls=0,additional_optimizer_calls=0,resource_mutations=0)
    return result'''
code = code.replace(anchor,
    "assert code.count('    return result')==1\ncode=code.replace('    return result',"
    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
