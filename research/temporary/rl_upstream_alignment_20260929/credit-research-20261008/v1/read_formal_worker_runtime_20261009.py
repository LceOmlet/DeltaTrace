"""Read loaded provenance/configuration through the owner's existing generic RPC.

No forward, state_dict, hooks, patches, optimizer calls or CUDA initialization.
The native actor queue chooses the boundary at which this read runs.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray


def inspect_worker(worker):
    from omegaconf import OmegaConf
    modules={}
    for name in ['verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos',
                 'verl.workers.fsdp_workers','verl.workers.sharding_manager.fsdp_vllm',
                 'deltatrace_rollout','reward_readout','qwen35_gdn_finite',
                 'qwen35_dense_finite_runner','accelerated.qwen35.native_fla_precision','vllm']:
        module=sys.modules.get(name)
        if module is None:
            modules[name]={'loaded':False}
        elif getattr(module,'__file__',None):
            p=Path(module.__file__)
            modules[name]=dict(loaded=True,path=str(p),resolved=str(p.resolve()),
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    owners=[]
    for name,owner in worker.worker_dict.items():
        if not getattr(owner,'_is_actor',False):continue
        config=owner.config
        keys=['model.path','model.lora_rank','model.lora_alpha',
              'actor.ppo_micro_batch_size_per_gpu','actor.ppo_mini_batch_size',
              'actor.ppo_epochs','actor.entropy_coeff','actor.clip_ratio_c',
              'rollout.max_model_len','rollout.max_num_seqs']
        owners.append(dict(name=name,rank=owner.rank,
                           configuration={key:OmegaConf.select(config,key) for key in keys}))
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),unix=time.time(),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),owners=owners,modules=modules)


if __name__=='__main__':
    output=Path(sys.argv[1]);address=sys.argv[2];prefix=sys.argv[3]
    ray.init(address=address,log_to_driver=False,ignore_reinit_error=False)
    named=[x for x in ray.util.list_named_actors(all_namespaces=True)
           if x['name'].startswith(prefix) and 'register_center' not in x['name']]
    assert len(named)==2,named
    refs=[ray.get_actor(x['name'],namespace=x['namespace']).execute_with_func_generator.remote(inspect_worker)
          for x in named]
    record=dict(started_unix=time.time(),query_pid=os.getpid(),query_birth=psutil.Process().create_time(),
                named_actors=named,driver_address=address,scope=__doc__,complete=False)
    output.write_text(json.dumps(record,indent=2)+'\n')
    try:
        results=ray.get(refs)
        record.update(complete=True,completed_unix=time.time(),results=results)
        output.write_text(json.dumps(record,indent=2)+'\n')
    finally:
        ray.shutdown()
