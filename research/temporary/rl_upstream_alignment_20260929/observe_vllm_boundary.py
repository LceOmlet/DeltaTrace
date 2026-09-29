"""Test-only Ray setup hook: inspect original return values without replacing calls."""
import json
import math
import os
from pathlib import Path
import sys
import time


def install():
    directory = Path(os.environ.get('DT_VLLM_OBSERVE_DIR', str(
        Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/upstream-alignment-20260929/probability-native-trainer')))

    def local_trace(frame, event, arg):
        if event != 'return' or arg is None:
            return local_trace
        rows = []
        for request in frame.f_locals.get('outputs', ()):
            for sample in request.outputs:
                values = [entry[token].logprob for token, entry in zip(sample.token_ids, sample.logprobs)]
                rows.append(dict(tokens=len(values), nonfinite=sum(not math.isfinite(v) for v in values),
                                 unique_tokens=len(set(sample.token_ids)), first_ids=sample.token_ids[:8]))
        record = dict(time_unix=time.time(), pid=os.getpid(), rows=rows)
        with (directory/f'raw-vllm-{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        return local_trace

    def trace(frame, event, arg):
        if (event == 'call' and frame.f_code.co_name == 'generate_sequences'
                and frame.f_code.co_filename.endswith('/vllm_rollout_spmd.py')):
            frame.f_trace_lines = False
            return local_trace
        return None

    sys.settrace(trace)
