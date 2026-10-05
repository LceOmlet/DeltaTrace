"""Read original CPU checkpoint artifacts; no model, optimizer restore or collective."""
import argparse
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time

import torch
from torch.distributed.tensor import DTensor


def resources():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if value.strip().split() and value.strip().split()[0].isdigit():
            values[key] = int(value.split()[0]) * 1024
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('VmRSS', 'VmHWM'):
            status[key] = int(value.split()[0]) * 1024
    return dict(mem_available_bytes=values['MemAvailable'],
                process_rss_bytes=status.get('VmRSS'),
                process_highwater_rss_bytes=status.get('VmHWM'),
                process_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def tensor_metadata(value):
    result = dict(type=type(value).__module__ + '.' + type(value).__qualname__,
                  shape=list(value.shape), dtype=str(value.dtype), requires_grad=value.requires_grad,
                  local_shape=list(local(value).shape), local_device=str(local(value).device))
    if isinstance(value, DTensor):
        result.update(placements=[str(p) for p in value.placements],
                      mesh=value.device_mesh.mesh.tolist(),
                      mesh_dim_names=value.device_mesh.mesh_dim_names)
    return result


def optimizer_steps(path):
    tick = time.perf_counter()
    original = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    result = dict(path=str(path), checkpoint_bytes=path.stat().st_size,
                  original_container_type=type(original).__name__, original_keys=list(original),
                  seconds=time.perf_counter() - tick)
    # Read original owner fields, without optimizer construction or state rewriting.
    states = original['state']
    groups = original['param_groups']
    result.update(state_count=len(states), param_group_count=len(groups),
                  param_group_sizes=[len(group['params']) for group in groups],
                  param_group_hyperparameters=[{key: value for key, value in group.items()
                                               if key != 'params'} for group in groups])
    counts = collections.Counter()
    step_types = collections.Counter()
    missing = []
    steps = []
    for parameter_id, state in states.items():
        if 'step' not in state:
            missing.append(str(parameter_id))
            continue
        step = state['step']
        if isinstance(step, torch.Tensor):
            native_step = local(step)
            step_types[str(step.dtype)] += 1
            value = native_step.item()
        else:
            step_types[type(step).__name__] += 1
            value = step
        counts[str(value)] += 1
        steps.append(dict(original_parameter_id=str(parameter_id), step=value))
    result.update(step_counter_distribution=dict(counts), step_value_types=dict(step_types),
                  missing_step_parameter_ids=missing, steps=steps,
                  interpretation='Original per-parameter optimizer counters; no inference from trainer iteration directory names.')
    return result


def compare_rank(root25, root100, rank):
    before = resources()
    tick = time.perf_counter()
    filename = f'model_world_size_2_rank_{rank}.pt'
    p25, p100 = root25 / filename, root100 / filename
    t = time.perf_counter()
    s25 = torch.load(p25, map_location='cpu', mmap=True, weights_only=False)
    s100 = torch.load(p100, map_location='cpu', mmap=True, weights_only=False)
    loaded = time.perf_counter() - t
    names25, names100 = set(s25), set(s100)
    result = dict(rank=rank, checkpoint25=str(p25), checkpoint100=str(p100),
                  checkpoint25_bytes=p25.stat().st_size, checkpoint100_bytes=p100.stat().st_size,
                  checkpoint25_keys=len(s25), checkpoint100_keys=len(s100),
                  keys_only25=sorted(names25 - names100), keys_only100=sorted(names100 - names25),
                  native_load_seconds=loaded, resources_before=before, resources_after_load=resources(),
                  frozen_base=dict(tensors=0, equal=0, local_elements=0, dtype_counts={}, different=[], metadata_mismatches=[]),
                  lora=dict(tensors=0, equal=0, different=0, local_elements=0,
                            changed_local_elements=0, maximum_absolute_difference=0.,
                            per_tensor=[], metadata_mismatches=[]))
    base_dtypes = collections.Counter()
    for index, name in enumerate(sorted(names25 & names100), 1):
        v25, v100 = s25[name], s100[name]
        a, b = local(v25), local(v100)
        m25, m100 = tensor_metadata(v25), tensor_metadata(v100)
        group = result['lora'] if '.lora_A.' in name or '.lora_B.' in name else result['frozen_base']
        group['tensors'] += 1
        group['local_elements'] += a.numel()
        if m25 != m100:
            group['metadata_mismatches'].append(dict(key=name, checkpoint25=m25, checkpoint100=m100))
        equal = torch.equal(a, b)
        group['equal'] += int(equal)
        if group is result['frozen_base']:
            base_dtypes[str(a.dtype)] += 1
            if not equal:
                group['different'].append(dict(key=name, checkpoint25=m25, checkpoint100=m100))
        else:
            group['different'] += int(not equal)
            # Native saved LoRA local tensors are FP32. Preserve their dtype.
            delta = b - a
            changed = int(torch.count_nonzero(delta).item())
            maximum = float(delta.abs().max().item()) if delta.numel() else 0.
            group['changed_local_elements'] += changed
            group['maximum_absolute_difference'] = max(group['maximum_absolute_difference'], maximum)
            group['per_tensor'].append(dict(key=name, equal=equal, checkpoint25=m25,
                checkpoint100=m100, changed_local_elements=changed,
                maximum_absolute_difference=maximum,
                l2_difference_native_dtype=float(torch.linalg.vector_norm(delta).item()),
                l2_checkpoint25_native_dtype=float(torch.linalg.vector_norm(a).item()),
                l2_checkpoint100_native_dtype=float(torch.linalg.vector_norm(b).item())))
        if index % 100 == 0:
            print(json.dumps(dict(phase='compare_native_locals', rank=rank, completed=index,
                                 total=len(names25 & names100), resources=resources())), flush=True)
    result['frozen_base']['dtype_counts'] = dict(base_dtypes)
    result['optimizer25'] = optimizer_steps(root25 / f'optim_world_size_2_rank_{rank}.pt')
    result['optimizer100'] = optimizer_steps(root100 / f'optim_world_size_2_rank_{rank}.pt')
    result['resources_after_comparison'] = resources()
    result['seconds'] = time.perf_counter() - tick
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint25', type=Path, required=True)
    parser.add_argument('--checkpoint100', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # CPU observation scheduling only; does not alter model/optimizer configuration.
    torch.set_num_threads(4)
    imports = [fingerprint(Path(torch.__file__)), fingerprint(Path(inspect.getfile(DTensor)))]
    receipt = dict(scope='Native CPU torch.load mmap/map_location=cpu/weights_only=False; native DTensor.to_local; exact frozen local value comparison and native dtype LoRA differences. No model or optimizer restore, gather, tensor dtype cast, CUDA initialization or training.',
                   pid=os.getpid(), started_unix=time.time(), script=fingerprint(Path(__file__)),
                   torch_version=torch.__version__, imported_sources=imports,
                   CPU_threads=torch.get_num_threads(),
                   CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   MACA_VISIBLE_DEVICES=os.environ.get('MACA_VISIBLE_DEVICES'),
                   process_group_initialized_before=torch.distributed.is_initialized(),
                   cuda_initialized_before=torch.cuda.is_initialized(), resources_before=resources(), ranks=[])
    print(json.dumps(dict(phase='begin_native_cpu_comparison', pid=receipt['pid'],
                         resources=receipt['resources_before'], imports=imports)), flush=True)
    with torch.no_grad():
        for rank in (0, 1):
            result = compare_rank(args.checkpoint25, args.checkpoint100, rank)
            receipt['ranks'].append(result)
            print(json.dumps(dict(phase='rank_complete', rank=rank,
                frozen_tensors=result['frozen_base']['tensors'],
                frozen_equal=result['frozen_base']['equal'],
                lora_changed_tensors=result['lora']['different'],
                lora_changed_elements=result['lora']['changed_local_elements'],
                lora_max_abs=result['lora']['maximum_absolute_difference'],
                optimizer25_step_counters=result['optimizer25']['step_counter_distribution'],
                optimizer100_step_counters=result['optimizer100']['step_counter_distribution'],
                resources=result['resources_after_comparison'])), flush=True)
            args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt.update(completed_unix=time.time(), resources_after=resources(),
                   process_group_initialized_after=torch.distributed.is_initialized(),
                   cuda_initialized_after=torch.cuda.is_initialized())
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(phase='native_cpu_comparison_complete', receipt=str(args.out),
                         sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                         resources=receipt['resources_after'],
                         process_group_initialized=receipt['process_group_initialized_after'],
                         cuda_initialized=receipt['cuda_initialized_after'])), flush=True)


if __name__ == '__main__':
    main()
