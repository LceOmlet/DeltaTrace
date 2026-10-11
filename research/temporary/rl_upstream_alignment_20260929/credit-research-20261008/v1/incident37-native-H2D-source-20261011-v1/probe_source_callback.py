"""Validate only MCPTI's ability to read an existing native H2D host source.

One 8KiB native Torch copy, no model/DT/PPO or numerical acceptance claim.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import resource
import time
import torch

root = Path(__file__).resolve().parent
lib = ctypes.CDLL(str(root / "memcpy_source_callback.so"))
lib.dt_source_start.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                               ctypes.c_uint64, ctypes.c_uint64]
lib.dt_source_start.restype = ctypes.c_int
lib.dt_source_stop.restype = ctypes.c_int
value = torch.arange(4096, dtype=torch.float32).to(torch.bfloat16).pin_memory()
expected = value.view(torch.uint8).numpy().tobytes()
path = root / "probe-source.jsonl"
start = time.perf_counter()
assert lib.dt_source_start(str(path).encode(), len(expected), 0, len(expected)) == 0
try:
    copied = value.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
finally:
    stop_status = lib.dt_source_stop()
rows = [json.loads(s) for s in path.read_text().splitlines()]
matches = [s for s in rows if s.get("src") == value.data_ptr()
           and s.get("dst") == copied.data_ptr()]
receipt = dict(unix=time.time(), stop_status=stop_status,
    seconds=time.perf_counter()-start, copy_bytes=len(expected),
    exact_native_source_rows=len(matches), sites=[s["site"] for s in matches],
    all_source_rows_exact=all(bytes.fromhex(s["CPU_row_hex"]) == expected for s in matches),
    selected=rows[-1], expected_sha256=hashlib.sha256(expected).hexdigest(),
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    production_changes=0, model_calls=0, DT_calls=0, numerical_tolerance=None,
    scope="Tool capability only; explicit synchronize belongs to this 8KiB probe, not the training observer")
(root / "probe-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
assert stop_status == 0 and len(matches) == 2 and receipt["all_source_rows_exact"]
assert set(receipt["sites"]) == {0, 1} and rows[-1]["dropped"] == 0
