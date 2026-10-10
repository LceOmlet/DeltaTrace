"""Read the existing native PyTorch trace; no model, CUDA, or production changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time

import ijson


def analyze(root, rank):
    meta = json.loads((root / f'rank{rank}-native-update-profile.json').read_bytes())
    trace = Path(meta['trace']['path'])
    weights = {h['inputs']['vocab_weights']['storage_ptr'] for h in meta['head_calls']}
    pointers = set(weights)
    for head in meta['head_calls']:
        pointers.update(t['storage_ptr'] for t in head['inputs'].values())
        pointers.update(t['storage_ptr'] for t in head['outputs'])
    categories, names, kernel_time = Counter(), Counter(), Counter()
    kernel_count, streams = Counter(), Counter()
    samples, scopes, memory = {}, [], []
    matched_memory, runtime_names = [], Counter()
    count = 0
    with trace.open('rb') as handle:
        for event in ijson.items(handle, 'traceEvents.item', use_float=True):
            count += 1
            cat, name = event.get('cat', ''), event.get('name', '')
            categories[cat] += 1
            names[(cat, name)] += 1
            if cat not in samples:
                samples[cat] = event
            if cat == 'kernel':
                kernel_time[name] += event.get('dur', 0)
                kernel_count[name] += 1
                streams[str(event.get('args', {}).get('stream'))] += 1
            if cat == 'cuda_runtime':
                runtime_names[name] += 1
            if 'FusedLinearForPPO' in name or name.startswith('FSDP'):
                scopes.append(event)
            if name == '[memory]':
                if len(memory) < 4:
                    memory.append(event)
                addr = event.get('args', {}).get('Addr')
                if addr in pointers:
                    matched_memory.append(event)
    digest = hashlib.sha256()
    with trace.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(rank=rank, trace=dict(path=str(trace), bytes=trace.stat().st_size,
                sha256=digest.hexdigest(), matches_recorded=digest.hexdigest() == meta['trace']['sha256']),
                events=count, categories=dict(categories), event_samples=samples,
                head_metadata=meta['head_calls'], head_weight_pointers=sorted(weights),
                memory_event_samples=memory, matching_pointer_memory=matched_memory,
                native_head_and_fsdp_scopes=scopes,
                kernels=[dict(name=name, count=kernel_count[name], sum_duration_us=duration)
                         for name, duration in kernel_time.most_common(30)],
                kernel_stream_counts=dict(streams), cuda_runtime_counts=dict(runtime_names),
                relevant_op_names=[dict(category=cat, name=name, count=n)
                    for (cat, name), n in names.items()
                    if any(s in name.lower() for s in ['fusedlinear', 'compiled', 'all_gather',
                                                      'free', 'resize', 'wait', 'stream'])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    started = time.time()
    report = dict(started_unix=started, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ranks=[analyze(args.result, rank) for rank in (0, 1)],
                  model_calls=0, production_changes=0,
                  scope='Finite profiled native update only; not evidence of failed-execution ordering')
    report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=report['elapsed_seconds'],
                         peak_RSS_bytes=report['peak_RSS_bytes'],
                         ranks=[dict(rank=r['rank'], events=r['events'], categories=r['categories'],
                                     head_scopes=len(r['native_head_and_fsdp_scopes']),
                                     matched_memory=len(r['matching_pointer_memory']),
                                     trace=r['trace']) for r in report['ranks']])))


if __name__ == '__main__':
    main()
