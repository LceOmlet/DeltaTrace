"""Check installed MCPTI address fields on the existing 8 KiB incident-row copy.

This is tool capability validation, not DT/PPO numerical acceptance. The
existing probe owns its copies and synchronization; this wrapper only subscribes
and compares the addresses it prints against the runtime's callback records.
"""
import contextlib
import ctypes
import hashlib
import io
import json
import pathlib
import resource
import runpy
import time

# The MCPTI subscriber attaches to a loaded runtime. Match its use in a live
# PyTorch worker rather than subscribing before PyTorch loads libmcruntime.
import torch

root = pathlib.Path(__file__).resolve().parent
probe = root.parent / "textcraft-native-MetaX-tracer-capability-20261011-v1/probe.py"
library = root / "memcpy_callback.so"
trace = root / "probe-callback.jsonl"
lib = ctypes.CDLL(str(library))
lib.dt_diagnostic_start.argtypes = [ctypes.c_char_p]
lib.dt_diagnostic_start.restype = ctypes.c_int
lib.dt_diagnostic_stop.restype = ctypes.c_int
start = time.perf_counter()
status = lib.dt_diagnostic_start(str(trace).encode())
if status != 0:
    raise RuntimeError(f"Official mcptiSubscribe/EnableCallback failed: {status}")
stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout):
        runpy.run_path(str(probe), run_name="__main__")
finally:
    stop_status = lib.dt_diagnostic_stop()
if stop_status != 0:
    raise RuntimeError(f"Official mcpti disable/unsubscribe or output failed: {stop_status}")
original = [json.loads(s) for s in stdout.getvalue().splitlines() if s.startswith("{")][-1]
rows = [json.loads(s) for s in trace.read_text().splitlines()]
summary, records = rows[0], rows[1:]
expected = [
    (original["CPU_source"], original["H2D_output"], original["bytes"]),
    (original["H2D_output"], original["D2D_output"], original["bytes"]),
    (original["D2D_output"], original["CPU_destination"], original["bytes"]),
]
matches = [
    [r for r in records if (r["src"], r["dst"], r["bytes"]) == item]
    for item in expected
]
receipt = {
    "unix": time.time(),
    "seconds": time.perf_counter() - start,
    "scope": "Official MCPTI callback capability on original 8KiB copy probe; no model or numerical acceptance",
    "original_probe": original,
    "original_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    "callback_summary": summary,
    "expected_address_matches": matches,
    "all_three_copies_have_exact_addresses": all(matches),
    "production_changes": 0,
    "model_calls": 0,
    "DT_calls": 0,
    "new_tolerance": None,
    "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
}
(root / "probe-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
if summary["dropped"] or not all(matches):
    raise RuntimeError("MCPTI callback does not cover the required copy addresses")
