"""Observe one real original VERL update on an already restored saved batch.

The caller owns complete checkpoint restoration, DataProto loading/cloning,
branch advantages and original compute_log_prob before/after observations.
This helper calls the supplied original worker.update_actor unchanged. Its two
instance proxies delegate real optimizer/scheduler steps, retaining their return
objects. It does not replace any loss, forward, backward, clip or offload method.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import time


ACTOR_SHA256 = "1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd"
CORE_SHA256 = "fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299"


def _source(value):
    """Report code and wrapped source; unwrapping never selects a callback."""
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    unwrapped = inspect.unwrap(function)
    filename = inspect.getsourcefile(unwrapped)
    path = Path(filename).resolve() if filename else None
    result = {
        "path": str(path) if path else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        "qualname": getattr(function, "__qualname__", None),
        "actual_callable_code_file": code.co_filename if code else None,
        "actual_callable_code_first_line": code.co_firstlineno if code else None,
    }
    return result


@contextmanager
def _instance_proxy(owner, name, proxy):
    existed = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, proxy)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def _local_cpu_copy(tensor):
    """Copy the actual native local storage, preserving dtype and placement."""
    from torch.distributed.tensor import DTensor
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    record = {"global_shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if isinstance(tensor, DTensor):
        record.update(
            placements=[str(p) for p in tensor.placements],
            mesh=tensor.device_mesh.mesh.detach().cpu().tolist(),
            mesh_device_type=tensor.device_mesh.device_type,
        )
        local = tensor.detach().to_local()
    else:
        record["placements"] = None
        local = tensor.detach()
    if isinstance(local, AsyncCollectiveTensor):
        local = local.wait()
    copied = local.to(device="cpu", copy=True)
    record.update(local_shape=list(copied.shape), local_elements=copied.numel(),
                  local_bytes=copied.numel() * copied.element_size())
    return copied, record


def _state_summary(optimizer):
    """Read real Adam state counters and native parameter-group settings."""
    counters, keys = Counter(), Counter()
    missing_steps = 0
    for state in optimizer.state.values():
        keys.update(state.keys())
        if "step" not in state:
            missing_steps += 1
            continue
        step = state["step"]
        if hasattr(step, "item"):
            step = step.item()
        counters[str(step)] += 1
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "parameter_count": len(group["params"]),
            **{key: group[key] for key in ("lr", "betas", "eps", "weight_decay",
                                           "amsgrad", "maximize", "foreach",
                                           "capturable", "differentiable", "fused")
               if key in group},
        })
    return {"optimizer_type": type(optimizer).__name__,
            "state_entries": len(optimizer.state), "state_key_counts": dict(keys),
            "step_counters": dict(counters), "missing_steps": missing_steps,
            "parameter_groups": groups}


def _resources():
    import psutil
    import torch

    process = psutil.Process()
    memory = process.memory_full_info()
    result = {
        "pid": process.pid, "pid_birth": process.create_time(),
        "rss_bytes": memory.rss, "pss_bytes": getattr(memory, "pss", None),
        "host_available_bytes": psutil.virtual_memory().available,
        "cuda_initialized": torch.cuda.is_initialized(),
        "distributed_initialized": torch.distributed.is_initialized(),
    }
    if result["cuda_initialized"]:
        result.update(cuda_device=torch.cuda.current_device(),
                      cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_max_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def observe_saved_native_adam_update(worker, data, *, original_update_actor,
                                     output_directory, branch):
    """Return the exact original worker output after one actual native update.

    Invoke on each native FSDP rank through the existing DP_COMPUTE_PROTO owner
    dispatch. ``data`` is its original local32 portion of the saved global64.
    ``original_update_actor`` must be the caller's bound super().update_actor.
    No stored old/ref logprob, advantage, mask or configuration is modified here.
    """
    import torch
    from verl.trainer.ppo import core_algos

    if branch not in ("DT", "GRPO", "dt", "grpo", "regularizers_only"):
        raise ValueError("Use one of the requested three saved-batch branches.")
    if getattr(original_update_actor, "__self__", None) is not worker:
        raise TypeError("Supply this worker's bound original super().update_actor.")
    actor = worker.actor
    actor_source = _source(type(actor))
    core_source = _source(core_algos.compute_policy_loss)
    if actor_source["sha256"] != ACTOR_SHA256 or core_source["sha256"] != CORE_SHA256:
        raise ValueError("This diagnostic targets the recorded original actor/core.")
    config = actor.config
    if (len(data) != 32 or config.ppo_mini_batch_size != 32
            or config.ppo_micro_batch_size_per_gpu != 4 or config.ppo_epochs != 1
            or config.use_dynamic_bsz):
        raise ValueError("Supply the recorded local32/B4/one-epoch native minibatch.")

    rank = int(worker.rank)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"rank{rank}-{branch}"
    phase_path = directory / (stem + "-phase.json")
    result_path = directory / (stem + "-native-adam.json")
    tensors_path = directory / (stem + "-native-parameter-shards.pt")
    optimizer = actor.actor_optimizer
    scheduler = worker.actor_lr_scheduler
    native_optimizer_step, native_scheduler_step = optimizer.step, scheduler.step
    named = {name: value for name, value in actor.actor_module.named_parameters()
             if value.requires_grad}
    before, before_layout, after, after_layout = {}, {}, {}, {}
    step_gradients, step_gradient_layout, step_records, scheduler_records = {}, {}, [], []
    report = {
        "scope": "One actual original worker update on one saved local32/global64 minibatch; original backward/clip/AdamW/scheduler/offload. No loss substitution or gradient gating.",
        "branch": branch, "rank": rank, "started_unix": time.time(),
        "sources": {"diagnostic": _source(observe_saved_native_adam_update),
                    "original_update_actor": _source(original_update_actor),
                    "actor": actor_source, "core": core_source,
                    "native_clip_and_step": _source(actor._optimizer_step),
                    "native_optimizer_step": _source(native_optimizer_step),
                    "native_scheduler_step": _source(native_scheduler_step)},
        "effective_actor_config": {
            key: getattr(config, key) for key in
            ("ppo_mini_batch_size", "ppo_micro_batch_size_per_gpu", "ppo_epochs",
             "use_dynamic_bsz", "entropy_coeff", "use_kl_loss", "kl_loss_coef",
             "kl_loss_type", "loss_agg_mode", "grad_clip")},
        "temperature": data.meta_info["temperature"],
        "multi_turn": data.meta_info.get("multi_turn", False),
        "input_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                         for key, value in data.batch.items()},
        "trainable_parameters": len(named), "resources_before": _resources(),
        "optimizer_before": _state_summary(optimizer),
        "scheduler_before": scheduler.state_dict(),
        "gradient_scope": "Raw native optimizer input gradient at the real AdamW.step entry, after the original actor clipping. None remains None. Only local shard storage is copied; no gather/cast/reducer or Adam formula.",
        "parameter_scope": "All requires_grad parameter local shards before/after original worker.update_actor. Original dtype; no full-model snapshot. Stored arrays allow exact storage-level parameter differences to be analyzed later.",
    }

    def phase(name):
        event = {"phase": name, "branch": branch, "rank": rank,
                 "unix": time.time(), **_resources()}
        phase_path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
        print("TEXTCRAFT_NATIVE_ADAM " + json.dumps(event), flush=True)

    @wraps(native_optimizer_step)
    def observed_optimizer_step(*args, **kwargs):
        phase("original_optimizer_step_enter")
        gradient, layout = {}, {}
        for name, parameter in actor.actor_module.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                gradient[name], layout[name] = None, {"grad_is_none": True}
            else:
                gradient[name], layout[name] = _local_cpu_copy(parameter.grad)
                layout[name]["grad_is_none"] = False
        step_gradients[str(len(step_records))] = gradient
        step_gradient_layout[str(len(step_records))] = layout
        record = {"entered_unix": time.time(), "optimizer_before": _state_summary(optimizer)}
        step_records.append(record)
        returned = native_optimizer_step(*args, **kwargs)
        record.update(returned_unix=time.time(), optimizer_after=_state_summary(optimizer))
        phase("original_optimizer_step_returned")
        return returned

    def native_scheduler_state():
        state = scheduler.state_dict()
        # LRScheduler.state_dict also includes instance attributes. Exclude only
        # this temporary observation function from its diagnostic state copy.
        if state.get("step") is observed_scheduler_step:
            del state["step"]
        return state

    @wraps(native_scheduler_step)
    def observed_scheduler_step(*args, **kwargs):
        record = {"entered_unix": time.time(), "before": native_scheduler_state()}
        scheduler_records.append(record)
        returned = native_scheduler_step(*args, **kwargs)
        record.update(returned_unix=time.time(), after=native_scheduler_state())
        return returned

    output = None
    error = None
    try:
        phase("copy_native_parameters_before")
        for name, parameter in named.items():
            before[name], before_layout[name] = _local_cpu_copy(parameter)
        phase("original_update_actor_begin")
        with _instance_proxy(optimizer, "step", observed_optimizer_step):
            with _instance_proxy(scheduler, "step", observed_scheduler_step):
                output = original_update_actor(data)
        report["original_metrics"] = output.meta_info.get("metrics", {})
        report["status"] = ("native_update_observed" if len(step_records) == 1
                            else "native_optimizer_step_count_differs_from_requested_one")
        phase("original_update_actor_returned")
    except BaseException as caught:
        error = caught
        report.update(status="original_update_or_observation_failed",
                      error_type=type(caught).__name__, error=str(caught))
    finally:
        # Both instance proxies have already restored the exact previous slots.
        for name, parameter in actor.actor_module.named_parameters():
            if parameter.requires_grad:
                after[name], after_layout[name] = _local_cpu_copy(parameter)
        payload = {"scope": report["parameter_scope"], "rank": rank, "branch": branch,
                   "before": before, "after": after,
                   "before_layout": before_layout, "after_layout": after_layout,
                   "optimizer_input_gradients_after_native_clip": step_gradients,
                   "gradient_layout": step_gradient_layout}
        torch.save(payload, tensors_path)
        with tensors_path.open("rb") as stream:
            tensor_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(
            parameter_shards={"path": str(tensors_path), "sha256": tensor_sha,
                              "bytes": tensors_path.stat().st_size},
            optimizer_step_calls=len(step_records), scheduler_step_calls=len(scheduler_records),
            optimizer_steps=step_records, scheduler_steps=scheduler_records,
            optimizer_after=_state_summary(optimizer), scheduler_after=scheduler.state_dict(),
            completed_unix=time.time(), resources_after=_resources(),
        )
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        phase("native_update_observation_saved")
    if error is not None:
        raise error
    return output
