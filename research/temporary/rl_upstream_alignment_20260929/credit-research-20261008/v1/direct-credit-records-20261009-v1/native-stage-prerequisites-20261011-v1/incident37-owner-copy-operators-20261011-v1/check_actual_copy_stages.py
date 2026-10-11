"""Localize saved corruption with unchanged owner copy/communication operators.

Actual two-rank root and layer0 parameters, no model forward or optimizer.
This supplements the incident; passing does not certify FSDP's full lifecycle.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from safetensors import safe_open
import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as owner


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    group = dist.distributed_c10d._get_default_group()
    args.output.mkdir(exist_ok=True, parents=True)
    report_path = args.output / f'rank{rank}.json'
    report = dict(unix=time.time(), pid=os.getpid(), rank=rank, source=identity(__file__),
                  owner=identity(owner.__file__), stages=[], phase='initialization',
                  model_calls=0, optimizer_steps=0, production_changes=0,
                  scope='Original copy operators on actual parameter values/shapes. No model or full FSDP lifecycle. Exact copy checks; no new floating tolerance.')
    assert report['owner']['sha256'] == '3bf079219682d1a9439b81754af65e87f69157ec69f20f065e5565519f7093d9'
    def save():
        report.update(unix=time.time(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    def check(label, actual, expected):
        start = time.perf_counter()
        equal = torch.equal(actual, expected)
        report['stages'].append(dict(label=label, equal=equal, elements=actual.numel(),
                                    seconds=time.perf_counter() - start))
        save()
        if not equal:
            # Only save differing rows and their positions, not another multi-GB copy.
            a, b = actual.reshape(-1), expected.reshape(-1)
            positions = (a != b).nonzero().flatten()
            artifact = args.output / f'rank{rank}-failure.pt'
            torch.save(dict(label=label, different_elements=positions.numel(),
                            first_positions=positions[:4096].cpu(),
                            actual=a[positions[:4096]].cpu(),
                            expected=b[positions[:4096]].cpu()), artifact)
            report['failure'] = identity(artifact)
            save()
            raise AssertionError(label)
    try:
        index = json.loads((args.model / 'model.safetensors.index.json').read_bytes())['weight_map']
        receipts = args.root / 'receipts'
        order_path = receipts / 'textcraft-native-incident37-layer0-payload-exact-20261011-v1.json'
        order = json.loads(order_path.read_bytes())['official_meta_order']
        reports = receipts / 'textcraft-native-incident37-native-flight-recorder-20261011-v1/result'
        incidents = []
        for r in (0, 1):
            meta = json.loads((reports / f'rank{r}.json').read_bytes())
            path = Path(meta['input_artifact']['path'])
            assert identity(path)['sha256'] == meta['input_artifact']['sha256']
            incidents.append(torch.load(path, map_location='cpu', weights_only=False, mmap=True))
        for module, entries in [('root', [('norm.weight', 'model.language_model.norm.weight'),
                                           ('lm_head.weight', 'lm_head.weight')]),
                                ('layer0', [(e['name'], None) for e in order])]:
            report['phase'] = module + '-load'
            save()
            local, expected = [], []
            for name, key in entries:
                if module == 'layer0' and '.lora_' in name:
                    train_key = 'base_model.model.model.layers.0.' + name
                    full = torch.cat([v['trainable_local_parameters'][train_key] for v in incidents], dim=0)
                else:
                    if key is None:
                        key = 'model.language_model.layers.0.' + name.replace('.base_layer.', '.')
                    with safe_open(args.model / index[key], framework='pt', device='cpu') as f:
                        full = f.get_tensor(key)
                assert full.shape[0] % 2 == 0
                full = full.to(torch.bfloat16)
                shard = full.chunk(2, dim=0)[rank].reshape(-1).pin_memory()
                local.append(shard.to(device, non_blocking=True))
                expected.append(full.to(device))
            split_sizes = [t.numel() for t in local]
            report[module] = dict(split_sizes=split_sizes, input_elements=sum(split_sizes),
                                  dtype=str(local[0].dtype))
            # Relevant owner test: v2.8.0 test_fully_shard_comm.py:_test_all_gather.
            # Separate stages deliberately synchronize checks to localize errors;
            # they do not replace or validate the full owner stream lifecycle.
            for repeat in range(4):
                report['phase'] = f'{module}-copy-in-{repeat}'
                packed, storage = torch.ops.fsdp.all_gather_copy_in(
                    local, split_sizes, sum(split_sizes), 2, rank,
                    torch.bfloat16, device, group.group_name, False)
                for i, (part, ref) in enumerate(zip(packed.split(split_sizes), local)):
                    check(f'{module}/{repeat}/copy-in/{i}', part, ref)
                report['phase'] = f'{module}-all-gather-{repeat}'
                dist.all_gather_into_tensor(storage, packed, group=group, async_op=False)
                gathered = storage.view(2, -1)
                for i, (part, ref) in enumerate(zip(gathered.split(split_sizes, dim=1), expected)):
                    check(f'{module}/{repeat}/all-gather/{i}', part.reshape(-1), ref.reshape(-1))
                report['phase'] = f'{module}-copy-out-{repeat}'
                outputs = [torch.empty_like(t).reshape(2, -1) for t in expected]
                torch.ops.fsdp.split_with_sizes_copy(gathered, split_sizes, dim=1, out=outputs)
                for i, (part, ref) in enumerate(zip(outputs, expected)):
                    check(f'{module}/{repeat}/copy-out/{i}', part.reshape(-1), ref.reshape(-1))
                del outputs, gathered, packed, storage
            del local, expected, full, shard
        report['phase'] = 'complete'
        save()
    except BaseException as e:
        report['error'] = repr(e)
        save()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
