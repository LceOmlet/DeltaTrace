"""Read resumed scalar state and actual imports through VERL's original RPC.

The actor queue chooses the boundary. No model call, state_dict, tensor copy,
GPU tensor read, synchronization, hook, patch or optimizer operation is added.
"""
import argparse
from collections import Counter
import hashlib
import json
import marshal
import os
from pathlib import Path
import time

import psutil
import ray
import read_formal_worker_runtime_20261009 as provenance


def inspect_worker(worker):
    import torch
    from types import CodeType
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner

    result = provenance.inspect_worker(worker)
    states = []
    for name, native in worker.worker_dict.items():
        if not getattr(native, '_is_actor', False):
            continue
        steps = Counter()
        unread_GPU_steps = 0
        for state in native.actor_optimizer.state.values():
            step = state.get('step')
            if isinstance(step, torch.Tensor):
                if step.device.type != 'cpu':
                    unread_GPU_steps += 1
                    continue
                step = step.item()
            steps[str(step)] += 1
        scheduler = native.actor_lr_scheduler
        states.append(dict(name=name, rank=native.rank,
            optimizer_step_counts=dict(steps), unread_GPU_steps=unread_GPU_steps,
            optimizer_group_lrs=[group['lr'] for group in native.actor_optimizer.param_groups],
            scheduler_last_epoch=scheduler.last_epoch,
            scheduler_step_count=scheduler._step_count,
            scheduler_last_lr=scheduler.get_last_lr()))
    function = owner.FSDPParamGroup.finalize_backward
    # marshal records reference sharing, so its bytes can differ after install
    # without any code change. Compare the code object to the pinned owner with
    # the same single name replacement; do not reinstall or modify the worker.
    compiled = compile(Path(owner.__file__).read_bytes(), owner.__file__, 'exec')
    cls = next(c for c in compiled.co_consts
               if isinstance(c, CodeType) and c.co_name == 'FSDPParamGroup')
    original = next(c for c in cls.co_consts
                    if isinstance(c, CodeType) and c.co_name == 'finalize_backward')
    expected = original.replace(co_names=tuple(
        'cuda' if name == 'accelerator' else name for name in original.co_names))
    result.update(resumed_states=states, effective_FSDP_function=dict(
        path=owner.__file__, file_SHA256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        code_SHA256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        instruction_bytes_SHA256=hashlib.sha256(function.__code__.co_code).hexdigest(),
        equals_compiled_owner_with_one_lookup_change=(function.__code__ == expected),
        instruction_bytes_equal_owner=(function.__code__.co_code == original.co_code),
        constants_equal_owner=(function.__code__.co_consts == original.co_consts),
        names=list(function.__code__.co_names)), model_calls=0, GPU_tensor_reads=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(Path(provenance.__file__).read_bytes()).hexdigest() == '7705cae717a0ad3f3b4ecb8a4c23e39bca9c8dcd5c22490294b9bfb8a62f6f5a'
    # Forward the existing inspector through Ray's serializer without adding
    # diagnostic modules to the formal workers' import path.
    ray.cloudpickle.register_pickle_by_value(provenance)
    ray.init(address=args.address, log_to_driver=False)
    named = [entry for entry in ray.util.list_named_actors(all_namespaces=True)
             if entry['name'].startswith(args.prefix) and 'register_center' not in entry['name']]
    assert len(named) == 2, named
    refs = [ray.get_actor(entry['name'], namespace=entry['namespace'])
            .execute_with_func_generator.remote(inspect_worker) for entry in named]
    receipt = dict(started_unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
                   address=args.address, named_actors=named, complete=False, scope=__doc__)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        receipt.update(results=ray.get(refs), complete=True, completed_unix=time.time())
        args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
