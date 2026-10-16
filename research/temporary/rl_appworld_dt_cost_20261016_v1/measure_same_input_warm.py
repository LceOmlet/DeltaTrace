"""One extra original DT readout on identical input and weights, before update.

This is an explicit cost intervention in the isolated trial, not formal
training. Preserve RNG and the primary return/report; no optimizer is called
between the first readout and its warm repeat.
"""
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-dt-cost-breakdown-20261016-v1'


def install_pair(worker):
    import hashlib,inspect,json,os,pickle,random,time
    from pathlib import Path
    import numpy as np
    import psutil,torch
    import reward_readout as owner
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    previous=json.loads((AUDIT/'nested-observation-installed.json').read_bytes())
    assert psutil.Process().create_time()=={v['pid']:v['birth'] for v in previous['workers']}[os.getpid()]
    if (AUDIT/f'rank{w.rank}-DT-cost.json').exists():return dict(rank=w.rank,status='not_installed_primary_already_completed')
    cls=owner.DirectActionTargetReadout;primary=cls.trajectories
    assert 'outer' in inspect.getclosurevars(primary).nonlocals
    def digest(value,h):
        h.update(type(value).__name__.encode())
        if isinstance(value,torch.Tensor):
            h.update(str((value.shape,value.dtype)).encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(value,np.ndarray):
            h.update(str((value.shape,value.dtype)).encode())
            if value.dtype.hasobject:
                for item in value.reshape(-1):digest(item,h)
            else:h.update(value.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):digest(key,h);digest(value[key],h)
        elif isinstance(value,(list,tuple)):
            for item in value:digest(item,h)
        else:h.update(pickle.dumps(value,protocol=4))
    def checksum(value):
        h=hashlib.sha256();digest(value,h);return h.hexdigest()
    def archive(label):
        for name in ['DT-cost','DT-partial','nested-components']:
            path=AUDIT/f'rank{w.rank}-{name}.json'
            if path.exists():(AUDIT/f'rank{w.rank}-{name}-{label}.json').write_bytes(path.read_bytes())
    def observed(instance,*args,**kwargs):
        scheduler=w.actor_lr_scheduler.last_epoch;optimizer_id=id(w.actor.actor_optimizer)
        initial_hash=checksum((args,kwargs));first=primary(instance,*args,**kwargs)
        primary_report=instance.last_report;archive('primary')
        assert checksum((args,kwargs))==initial_hash
        py,np_state=random.getstate(),np.random.get_state()
        cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
        record=dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),input_SHA256=initial_hash,
            started_unix=time.time(),scheduler_before=scheduler,optimizer_identity_before=optimizer_id,
            additional_original_DT_readouts_per_rank=1,optimizer_updates_between_readouts=0,
            cold_warm_same_weight_input=True)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                second=primary(instance,*args,**kwargs)
            archive('warm');del second
            record.update(status='same_input_native_warm_repeat_completed',input_SHA256_after=checksum((args,kwargs)))
        finally:
            random.setstate(py);np.random.set_state(np_state);instance.last_report=primary_report
            record.update(completed_unix=time.time(),scheduler_after=w.actor_lr_scheduler.last_epoch,
                optimizer_identity_after=id(w.actor.actor_optimizer),
                RNG_restored=torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
                    and random.getstate()==py and checksum(np.random.get_state())==checksum(np_state),
                primary_report_restored=instance.last_report is primary_report,
                primary_return_object_preserved=True)
            (AUDIT/f'rank{w.rank}-same-input-pair.json').write_text(json.dumps(record,indent=2)+'\n')
        assert record['input_SHA256_after']==initial_hash and record['RNG_restored']
        assert record['scheduler_after']==scheduler and record['optimizer_identity_after']==optimizer_id
        return first
    cls.trajectories=observed
    return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),status='same_input_native_cost_pair_installed')


if __name__=='__main__':
    import psutil,ray
    job=json.loads((AUDIT/'job.json').read_bytes());p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
    gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2
        values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install_pair) for a in names],timeout=1200)
        (AUDIT/'same-input-pair-installed.json').write_text(json.dumps(dict(unix=time.time(),workers=values),indent=2)+'\n')
    finally:ray.shutdown()
