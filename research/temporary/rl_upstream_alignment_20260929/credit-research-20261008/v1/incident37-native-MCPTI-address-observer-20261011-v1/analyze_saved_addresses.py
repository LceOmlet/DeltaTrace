"""Read-only CPU analysis of the one completed native MCPTI incident replay.

Address overlaps are spatial observations, not proof of a live-storage write or
GPU ordering. A finite run cannot establish the failed run's transport state.
"""
import collections
import hashlib
import json
import pathlib
import resource
import time

import torch


ROOT = pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
JOB = ROOT / 'receipts/textcraft-native-incident37-MCPTI-addresses-20261011-v1'
NORMAL_ROW_SHA = '4defedf718a10ac463b43d89a6b0b0f9d5e98ec0a459eeab066f8673b125423a'
VOCAB_ID = 159048


def sha_tensor(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def intersects(pointer, length, start, size):
    return pointer < start + size and pointer + length > start


def main():
    started = time.perf_counter()
    result = {'unix': time.time(), 'ranks': [], 'GPU_calls': 0,
              'model_calls': 0, 'DT_calls': 0, 'production_changes': 0,
              'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    for rank in (0, 1):
        folder = JOB / 'result'
        report = json.loads((folder / f'rank{rank}-native-memcpy-addresses.json').read_text())
        lines = (folder / f'rank{rank}-native-memcpy-addresses.jsonl').read_text().splitlines()
        summary = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
        entries = [r for r in records if r['site'] == 0]
        exits = [r for r in records if r['site'] == 1]
        gemm = torch.load(folder / f'rank{rank}-native-GEMM-rows.pt',
                          map_location='cpu', weights_only=False, mmap=True)
        weights = gemm['competing_weights']
        row_sha = sha_tensor(weights[0])
        sources = report['actual_flat_source_metadata_before_update']
        cpu_ranges = []
        for source in sources:
            pointer, size = source['pointer'], source['bytes']
            reads = [r for r in entries if intersects(r['src'], r['bytes'], pointer, size)]
            writes = [r for r in entries if intersects(r['dst'], r['bytes'], pointer, size)]
            cpu_ranges.append(dict(metadata=source, retained_read_calls=len(reads),
                retained_write_calls=len(writes), read_calls=reads, write_calls=writes))
        gpu_ranges = []
        for pointer in sorted({r['right_pointer'] for r in gemm['metadata']['mm_calls']}):
            shape = gemm['metadata']['mm_calls'][0]['right_shape']
            size = shape[0] * shape[1] * weights.element_size()
            row_pointer = pointer + VOCAB_ID * shape[0] * weights.element_size()
            reads = [r for r in entries if r['src'] == row_pointer
                     and r['bytes'] == weights.shape[1] * weights.element_size()]
            writes = [r for r in entries if intersects(r['dst'], r['bytes'], pointer, size)]
            gpu_ranges.append(dict(pointer=pointer, bytes=size, selected_row_pointer=row_pointer,
                retained_exact_selected_row_D2H_reads=len(reads),
                retained_spatial_destination_overlaps=len(writes), destination_overlaps=writes))
        post = torch.load(folder / f'rank{rank}-postfailure-shard.pt',
                          map_location='cpu', weights_only=False)
        cpu_rows = [dict(parameter=r['parameter'], sha256=sha_tensor(r['buffer']),
                        exact_normal_row=sha_tensor(r['buffer']) == NORMAL_ROW_SHA)
                    for r in post if 'buffer' in r]
        loss = torch.load(folder / f'rank{rank}-loss-inputs-microbatch1.pt',
                          map_location='cpu', weights_only=False)
        runtime = json.loads((folder / f'rank{rank}.json').read_text())
        result['ranks'].append(dict(rank=rank, callback_summary=summary,
            retained_entries=len(entries), retained_exits=len(exits),
            retained_copy_kinds=dict(collections.Counter(r['kind'] for r in entries)),
            retained_nonzero_exit_status=[r for r in exits if r['result'] != 0],
            callback_recording_complete=summary['dropped'] == 0,
            finite_microbatches=[r['gradient_finite'] for r in runtime['microbatches']],
            actual_optimizer_steps=runtime['actual_optimizer_steps'],
            actual_GEMM_competing_rows=weights.shape[0], first_actual_GEMM_row_sha256=row_sha,
            first_actual_GEMM_row_exact_normal=row_sha == NORMAL_ROW_SHA,
            all_actual_GEMM_rows_exact_first=bool(torch.equal(weights, weights[0].expand_as(weights))),
            post_exit_CPU_rows=cpu_rows,
            loss_input_finiteness={k:bool(torch.isfinite(v).all()) for k,v in loss.items()
                                  if isinstance(v, torch.Tensor)},
            CPU_source_ranges=cpu_ranges, GPU_weight_ranges=gpu_ranges))
        del gemm, weights, post, loss
    result.update(seconds=time.perf_counter() - started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  CUDA_initialized=torch.cuda.is_initialized(),
                  limitations=['Tail callback records were dropped; no complete-update trace claim.',
                    'All microbatches finite; these addresses do not explain the previous failed execution.',
                    'Only three runtime memcpy API variants observed; kernel writes not observed.',
                    'Spatial address reuse does not prove simultaneous live-storage overlap.',
                    'Host API order is not GPU completion order.'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
