"""Dump the installed PyTorch recorder at the existing terminal report seam.

Reuse the previous post-report observer: dump native records, then read the
CPU frozen source with its original observer. No new actor class or pre-report hook.
"""
from pathlib import Path
import hashlib
import json
import os
import pickle
import time

import after_native_report_cpu_source as seam

SEAM_SHA = '251ca219fba4c1479dda275e633a2a56971b89f9d1a67928848df2af306e7759'
ORIGINAL_CPU_SOURCE_CAPTURE = seam.capture

def capture(actor, original_report_path):
    import torch
    output = Path(os.environ['DT_INCIDENT_AFTER_REPORT_RESULT'])
    raw = torch._C._distributed_c10d._dump_nccl_trace(
        includeCollectives=True, includeStackTraces=False, onlyActive=False)
    path = output / f'rank{actor.rank}-native-flight-recorder.pkl'
    path.write_bytes(raw)
    trace = pickle.loads(raw)
    json_path = path.with_suffix('.json')
    json_path.write_bytes((json.dumps(trace, indent=2) + '\n').encode())
    entries = trace.get('entries', [])
    report = dict(rank=actor.rank, unix=time.time(), source=__file__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_terminal_report=str(original_report_path),
        api='torch._C._distributed_c10d._dump_nccl_trace',
        options=dict(includeCollectives=True, includeStackTraces=False, onlyActive=False),
        native_environment={k:os.environ.get(k) for k in (
            'TORCH_NCCL_TRACE_BUFFER_SIZE','TORCH_NCCL_TRACE_CPP_STACK','TORCH_NCCL_ENABLE_TIMING')},
        trace_version=trace.get('version'), entry_count=len(entries),
        first_entry=entries[0] if entries else None, last_entry=entries[-1] if entries else None,
        native_trace=dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        JSON_trace=dict(path=str(json_path), bytes=json_path.stat().st_size,
            sha256=hashlib.sha256(json_path.read_bytes()).hexdigest()),
        pre_terminal_report_added_tensor_operations=0,
        pre_terminal_report_added_synchronizations=0,
        added_model_calls=0, production_changes=0,
        limitation='Native recorder adds CPU-side metadata overhead; a finite replay is not a repair.')
    seam.ORIGINAL_WRITE_TEXT(output / f'rank{actor.rank}-native-flight-recorder-report.json',
        json.dumps(report, indent=2) + '\n')
    # Original CPU-only source observer runs strictly after the terminal report
    # and native recorder dump. No pre-failure tensor operation is added.
    ORIGINAL_CPU_SOURCE_CAPTURE(actor, original_report_path)

def install():
    assert hashlib.sha256(Path(seam.__file__).read_bytes()).hexdigest() == SEAM_SHA
    seam.capture = capture
    seam.install()
