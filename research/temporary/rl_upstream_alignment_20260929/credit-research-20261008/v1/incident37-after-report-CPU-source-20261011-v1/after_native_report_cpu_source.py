"""Diagnostic-only capture after the original GEMM report has been written.

No actor class, model operation, FSDP transition, GPU copy or pre-failure
observation is added. pathlib report emission is the existing boundary.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ORIGINAL_WRITE_TEXT = Path.write_text
RESULT = os.environ.get("DT_INCIDENT_AFTER_REPORT_RESULT", "")
DONE = set()


def capture(actor, original_report_path):
    import torch
    observations, errors = [], []
    try:
        names = {id(m): n for n, m in actor.actor_module_fsdp.named_modules()}
        for module in actor.actor_module_fsdp.modules():
            get_state = getattr(module, "_get_fsdp_state", None)
            if get_state is None:
                continue
            group = get_state()._fsdp_param_group
            if group is None:
                continue
            for p in group.fsdp_params:
                info = p._module_info
                name = names.get(id(info.module), "")
                if not name.endswith("lm_head") or info.param_name != "weight":
                    continue
                shard_rank = p.mesh_info.shard_mesh_rank
                lo = shard_rank * p.padded_sharded_param_size[0]
                full = getattr(p, "_unsharded_param", None)
                row = dict(parameter=name + ".weight", shard_rank=shard_rank,
                    shard_start=lo, sharded_state=str(p.sharded_state),
                    group_training_state=str(group._training_state),
                    all_gather_result_pending=group._all_gather_result is not None,
                    full_storage_bytes=full.untyped_storage().nbytes() if full is not None else None,
                    CPU_source_rows=[])
                observations.append(row)
                for source_name in ["_sharded_param_data", "_sharded_local_tensor"]:
                    value = getattr(p, source_name, None)
                    if value is None:
                        continue
                    source = dict(source=source_name, device=str(value.device),
                        shape=list(value.shape), dtype=str(value.dtype), pointer=value.data_ptr())
                    row["CPU_source_rows"].append(source)
                    if value.device.type != "cpu":
                        source["not_captured"] = "CPU source unavailable; no GPU read"
                        continue
                    local_row = 159048 - lo
                    if not 0 <= local_row < p.padded_sharded_param_size[0]:
                        source["not_owned_on_this_rank"] = True
                        continue
                    width = p._orig_size[-1]
                    source["local_row"] = local_row
                    source["buffer"] = value.reshape(-1)[local_row*width:(local_row+1)*width].detach().clone()
    except Exception as error:
        errors.append(repr(error))
    output = Path(RESULT)
    path = output / f"rank{actor.rank}-after-native-report-CPU-source.pt"
    torch.save(observations, path)
    report = dict(rank=actor.rank, unix=time.time(), source=__file__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_report=str(original_report_path), errors=errors,
        observations=[dict(r, CPU_source_rows=[{k:v for k,v in s.items() if k != "buffer"}
            for s in r["CPU_source_rows"]]) for r in observations],
        artifact=dict(path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        pre_original_report_value_reads=0, GPU_calls=0, added_synchronizations=0,
        model_calls=0, optimizer_steps=0, production_changes=0)
    ORIGINAL_WRITE_TEXT(output / f"rank{actor.rank}-after-native-report-CPU-source.json",
        json.dumps(report, indent=2) + "\n")


def install():
    if getattr(Path.write_text, "_dt_after_report_observer", False):
        return
    @functools.wraps(ORIGINAL_WRITE_TEXT)
    def write_text(self, data, *args, **kwargs):
        result = ORIGINAL_WRITE_TEXT(self, data, *args, **kwargs)
        if RESULT and str(self.parent) == RESULT and self.name in {
                "rank0-native-GEMM-capture.json", "rank1-native-GEMM-capture.json"} and self.name not in DONE:
            DONE.add(self.name)
            try:
                frame = sys._getframe(1)
                actor = frame.f_locals.get("self")
                if frame.f_code.co_name != "update_actor" or actor is None:
                    raise RuntimeError("Original GEMM report caller not available")
                capture(actor, self)
            except Exception as error:
                ORIGINAL_WRITE_TEXT(Path(RESULT) / f"pid{os.getpid()}-after-report-error.json",
                    json.dumps(dict(error=repr(error), original_report=str(self))) + "\n")
        return result
    write_text._dt_after_report_observer = True
    Path.write_text = write_text
    directory = os.environ.get("DT_INCIDENT_AFTER_REPORT_HOOK_RECORD")
    if directory:
        p = Path(directory) / f"pid{os.getpid()}-hook.json"
        ORIGINAL_WRITE_TEXT(p, json.dumps(dict(pid=os.getpid(), unix=time.time(),
            source=__file__, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            callback="Only after original native GEMM report write returned", GPU_calls=0)) + "\n")
