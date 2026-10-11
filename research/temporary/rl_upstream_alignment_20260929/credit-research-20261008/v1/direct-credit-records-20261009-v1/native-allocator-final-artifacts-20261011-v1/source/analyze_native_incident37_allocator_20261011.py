"""Read the native allocator snapshot for the actually observed head address.

This reports original allocation/free events and stacks. An overlapping
allocation in a completed CPU trace alone does not prove a GPU lifetime race.
No model/GPU execution, numerical tolerance, or production change is involved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time

import torch


def contains(address, size, point):
    return address <= point < address + size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    report = dict(unix=time.time(), source=str(Path(__file__)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        GPU_calls=0, model_calls=0, DT_calls=0, production_changes=0,
        new_numerical_tolerance=None, ranks=[])
    for rank in range(2):
        row = dict(rank=rank, unavailable=[])
        report['ranks'].append(row)
        result = args.job / 'result'
        state = json.loads((result / f'rank{rank}.json').read_bytes())
        row.update(phase=state['phase'], optimizer_steps=state['actual_optimizer_steps'])
        snapshot_path = result / f'rank{rank}-native-allocator.pickle'
        gemm_path = result / f'rank{rank}-native-GEMM-rows.pt'
        if not snapshot_path.is_file() or not gemm_path.is_file():
            row['unavailable'] = [str(p) for p in [snapshot_path, gemm_path] if not p.is_file()]
            continue
        with snapshot_path.open('rb') as stream:
            snapshot = pickle.load(stream)
        gemm = torch.load(gemm_path, map_location='cpu', weights_only=False, mmap=True)
        metadata = gemm['metadata']
        pointers = sorted(set(call['right_pointer'] for call in metadata['mm_calls']))
        width = metadata['selection']['hidden_dim']
        token = metadata['competing_vocab_id']
        itemsize = gemm['competing_weights'].element_size()
        row.update(snapshot_bytes=snapshot_path.stat().st_size,
            snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            native_mm_calls=len(metadata['mm_calls']), W_pointers=pointers,
            actual_weight_addresses=[p + token * width * itemsize for p in pointers],
            observed_weight_rows=len(metadata['copied_indices']), address_histories=[])
        indices = metadata['copied_indices']
        if indices:
            weights = gemm['competing_weights'][indices]
            row['actual_weight_rows_all_equal'] = bool(torch.equal(weights, weights[0].expand_as(weights)))
            row['actual_weight_row_sha256'] = hashlib.sha256(weights[0].view(torch.uint8).numpy().tobytes()).hexdigest()
        row['trace_counts'] = [dict(events=len(events), actions=dict(Counter(e['action'] for e in events)))
            for events in snapshot['device_traces']]
        for pointer in row['actual_weight_addresses']:
            history = dict(row_pointer=pointer, events=[], current_blocks=[])
            row['address_histories'].append(history)
            for device, events in enumerate(snapshot['device_traces']):
                for index, event in enumerate(events):
                    address = event.get('addr')
                    size = event.get('size')
                    if address is not None and size is not None and contains(address, size, pointer):
                        history['events'].append(dict(device=device, trace_index=index, **event))
            for segment in snapshot['segments']:
                for block in segment['blocks']:
                    if contains(block['address'], block['size'], pointer):
                        history['current_blocks'].append(dict(device=segment['device'],
                            segment_address=segment['address'], segment_stream=segment['stream'], **block))
        loss_path = result / f'rank{rank}-loss-inputs-microbatch1.pt'
        if loss_path.is_file():
            loss = torch.load(loss_path, map_location='cpu', weights_only=False)
            row['logp_gradient_nan_count'] = int(torch.isnan(loss['log_prob_gradient']).sum())
        del snapshot, gemm
    report.update(seconds=time.perf_counter() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized(),
        limitation='Allocator CPU order and stack ownership do not alone establish ordering of asynchronous GPU reads/writes.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
