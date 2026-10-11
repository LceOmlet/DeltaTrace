"""Ray's documented worker_process_setup_hook; diagnostics only, no model hooks.

The 1,017,118,720-byte BF16 shard and 124160x4096 padded shape were captured
from the actual native FSDP head. The offset is row159048 in rank1's shard.
Embedding shards can have the same size; identify sources from saved values,
not just their byte count. Callback exit does not imply async GPU completion.
"""
import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time


def setup():
    # Installed MCPTI must attach after Torch has loaded the native runtime.
    import torch
    import psutil

    root = Path(os.environ["DT_DIAGNOSTIC_SOURCE_DIR"])
    output = Path(os.environ["DT_DIAGNOSTIC_SOURCE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    library = root / "memcpy_source_callback.so"
    actual_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    assert actual_sha == os.environ["DT_DIAGNOSTIC_SOURCE_LIBRARY_SHA"]
    lib = ctypes.CDLL(str(library))
    lib.dt_source_start.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                                   ctypes.c_uint64, ctypes.c_uint64]
    lib.dt_source_start.restype = ctypes.c_int
    lib.dt_source_stop.restype = ctypes.c_int
    path = output / f"pid{os.getpid()}-native-H2D-source.jsonl"
    status = lib.dt_source_start(str(path).encode(), 124160 * 4096 * 2,
                                (159048 - 124160) * 4096 * 2, 4096 * 2)
    if status != 0:
        raise RuntimeError(f"Installed MCPTI source subscriber failed: {status}")
    # Keep library alive; no replacement of native actor or update method.
    globals()["_diagnostic_library"] = lib
    atexit.register(lib.dt_source_stop)
    (output / f"pid{os.getpid()}-worker-setup.json").write_text(json.dumps(dict(
        unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
        source=__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library=str(library), library_sha256=actual_sha,
        torch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
        official_hook="Ray RuntimeEnv.worker_process_setup_hook", start_status=status,
        scope="Read-only host source snapshots at native H2D API entry/exit",
        added_GPU_value_reads=0, added_GPU_copies=0, added_synchronizations=0,
        production_changes=0, numerical_tolerance=None), indent=2) + "\n")
