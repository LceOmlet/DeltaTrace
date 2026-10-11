"""Pinned FSDP CUDA-event compatibility fix, activated only by a Ray setup hook.

In torch 2.8.0+metax3.5.3.9, the generic Stream.wait_event does not wait on
the CUDA Event produced by FSDP. finalize_backward then releases an outstanding
prefetch buffer too early. Keep the installed owner function and change its
single accelerator lookup to cuda. No owner body is copied and no installed
package or global torch API is modified. See the incident37 verification receipt.
"""

import hashlib
import json
import marshal
import os
from pathlib import Path
import time

OWNER_SHA256 = "2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5"
_receipt = None


def install():
    """Install the tested one-lookup fix in this process's original owner."""
    global _receipt
    if _receipt is not None:
        return dict(_receipt)

    import torch
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner

    if torch.__version__ != "2.8.0+metax3.5.3.9":
        raise RuntimeError("FSDP event fix is verified only for torch 2.8.0+metax3.5.3.9")
    owner_path = Path(owner.__file__)
    if hashlib.sha256(owner_path.read_bytes()).hexdigest() != OWNER_SHA256:
        raise RuntimeError("Installed FSDP owner differs from the verified owner")
    function = owner.FSDPParamGroup.finalize_backward
    original = function.__code__
    if original.co_names.count("accelerator") != 1 or "cuda" in original.co_names:
        raise RuntimeError("Unexpected FSDP finalize_backward code; refusing an unverified override")
    names = tuple("cuda" if name == "accelerator" else name for name in original.co_names)
    patched = original.replace(co_names=names)
    assert patched.co_code == original.co_code and patched.co_consts == original.co_consts
    function.__code__ = patched
    _receipt = dict(
        owner_path=str(owner_path), owner_file_SHA256=OWNER_SHA256,
        function="FSDPParamGroup.finalize_backward",
        original_code_SHA256=hashlib.sha256(marshal.dumps(original)).hexdigest(),
        patched_code_SHA256=hashlib.sha256(marshal.dumps(patched)).hexdigest(),
        original_names=list(original.co_names), patched_names=list(patched.co_names),
        instruction_bytes_unchanged=True, constants_unchanged=True,
        copied_function_bodies=0, installed_files_modified=0,
        numerical_or_algorithm_changes=0,
    )
    return dict(_receipt)


def worker_setup():
    """Ray's native worker_process_setup_hook; no CUDA initialization or model load."""
    import psutil
    import torch

    receipt = install()
    receipt.update(
        pid=os.getpid(), birth=psutil.Process().create_time(), unix=time.time(),
        source=str(Path(__file__).resolve()),
        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_initialized=torch.cuda.is_initialized(),
        activation="Ray runtime_env.worker_process_setup_hook",
    )
    destination = Path(os.environ["DT_FSDP_EVENT_WAIT_RECEIPTS"])
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"worker-{receipt['pid']}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
