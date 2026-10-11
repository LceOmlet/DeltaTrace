"""Pass a documented Ray startup hook to the unchanged original failing entry.

No new actor class, training method, model path, optimizer, loss, or DT logic.
This is a diagnostic launcher outside all production launch paths.
"""
import functools
import hashlib
import os
from pathlib import Path
import runpy
import sys

import ray

ROOT = Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922")
ORIGINAL = ROOT / "receipts/textcraft-native-incident37-competing-weight-capture-20261011-v1/source/capture_native_incident37_GEMM_rows_20261011.py"
ORIGINAL_SHA = "35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8"
native_init = ray.init


@functools.wraps(native_init)
def init_with_official_setup_hook(*args, **kwargs):
    runtime_env = dict(kwargs.get("runtime_env") or {})
    assert "worker_process_setup_hook" not in runtime_env
    runtime_env["worker_process_setup_hook"] = "incident37_mcpti_source_setup.setup"
    kwargs["runtime_env"] = runtime_env
    return native_init(*args, **kwargs)


def main():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    assert os.environ["DT_DIAGNOSTIC_SOURCE_OUTPUT"]
    sys.path.insert(0, str(ORIGINAL.parent))
    sys.argv[0] = str(ORIGINAL)
    ray.init = init_with_official_setup_hook
    try:
        runpy.run_path(str(ORIGINAL), run_name="__main__")
    finally:
        ray.init = native_init


if __name__ == "__main__":
    main()
