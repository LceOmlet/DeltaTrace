"""Check the time coverage of an existing official allocator snapshot."""
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import resource
import time

import torch


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    job = root / 'receipts/textcraft-native-incident37-allocator-20261011-v1'
    result = job / 'result'
    report = dict(unix=time.time(), GPU_calls=0, model_calls=0, production_changes=0, ranks=[])
    for rank in (0, 1):
        path = result / f'rank{rank}-native-allocator.pickle'
        phases = [json.loads(line) for line in
                  (result / f'rank{rank}-phases.jsonl').read_text().splitlines()]
        first = next(p for p in phases if p.get('microbatch') == 0)
        second = next(p for p in phases if p.get('microbatch') == 1)
        # The selected head was called between the first loss gradient and
        # the second loss gradient. No later reuse of its address establishes
        # ownership during this earlier call.
        bracket = [first['unix'], second['unix']]
        snapshot = pickle.loads(path.read_bytes())
        row = dict(rank=rank, snapshot=dict(path=str(path), bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                   selected_second_head_time_bracket=bracket, devices=[])
        for device, events in enumerate(snapshot['device_traces']):
            times = [e['time_us'] / 1e6 for e in events if 'time_us' in e]
            row['devices'].append(dict(device=device, count=len(events),
                first_time=times[0] if times else None,
                last_time=times[-1] if times else None,
                chronological=all(a <= b for a, b in zip(times, times[1:])),
                events_inside_selected_call_bracket=sum(bracket[0] <= t <= bracket[1] for t in times),
                events_before_bracket=sum(t < bracket[0] for t in times),
                events_after_bracket=sum(t > bracket[1] for t in times)))
        report['ranks'].append(row)
        del snapshot
    owner = Path(inspect.getsourcefile(torch.cuda.memory._record_memory_history))
    report.update(owner_path=str(owner), owner_sha256=hashlib.sha256(owner.read_bytes()).hexdigest(),
        owner_docstring=torch.cuda.memory._record_memory_history.__doc__,
        seconds=time.perf_counter()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized())
    output = root / 'receipts/textcraft-native-incident37-retained-allocator-window-20261011-v1.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
