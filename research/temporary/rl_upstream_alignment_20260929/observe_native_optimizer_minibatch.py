"""Isolated observation of one complete original actor optimizer minibatch.

The caller supplies the unmodified local 32-row DataProto plus the externally
computed official GRPO advantages in ``diagnostic_grpo_advantages``. The real
worker owns load/shard/offload. This module is not wired into any production
entry point. The caller must also temporarily disable/restore the worker's
scheduler step: the original worker executes it after update_policy returns.

Every pass calls the original class update_policy. Scalar backward hooks select
one original loss component; native forward, mask, accumulation, backward and
gradient clipping remain the owner's calls. Only optimizer.step is a no-op.
This is diagnostic execution, not evidence that an original training update
passed. Zero upstream gradients do not prune other branches: nonfinite backward
intermediates can still affect a selected component and must remain visible.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import random
import time

from observe_native_actor_loss_gradients import (
    _source_identity, capture_native_gradient, native_gradient_statistics,
)


ACTOR_SHA256 = "1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd"
CORE_SHA256 = "fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299"
LABELS = ("dt_pg", "weighted_entropy", "weighted_kl", "grpo_pg")


@contextmanager
def _temporary_attribute(owner, name, value):
    """Restore an instance/module attribute, including its original absence."""
    existed = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


class NativeLossHooks:
    """Observe/gate only the actual actor's three existing scalar outputs.

    The pinned vanilla PG function calls its *core module* agg_loss. The actor
    module's agg_loss alias is used directly only for H then KL, after PG.
    This helper does not implement a policy loss or aggregate token losses.
    """

    def __init__(self, actor_owner, *, label, entropy_coeff, use_kl_loss, kl_coef,
                 policy_component=False):
        self.owner = actor_owner
        self.label = label
        self.entropy_coeff = entropy_coeff
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.policy_component = policy_component
        self.policy_loss = actor_owner.compute_policy_loss
        self.aggregate = actor_owner.agg_loss
        self.records = []
        self.pending = []
        self.handles = []

    def _scalar(self, value, component, coefficient):
        import torch

        selected = component == self.label or (component == "dt_pg" and
                   (self.label == "grpo_pg" or self.policy_component))
        self.records[-1][component] = {
            "value": value.detach().item(), "dtype": str(value.dtype),
            "native_coefficient": coefficient, "selected_backward": selected,
            "requires_grad": value.requires_grad,
        }
        if value.requires_grad:
            self.handles.append(value.register_hook(
                lambda gradient: gradient if selected else torch.zeros_like(gradient)))
        return value

    def compute_policy_loss(self, *args, **kwargs):
        if self.pending:
            raise RuntimeError("The observed original H/KL scalar calls did not finish before the next PG call.")
        result = self.policy_loss(*args, **kwargs)
        self.records.append({"microbatch_index": len(self.records)})
        self._scalar(result[0], "dt_pg", 1.0)
        self.records[-1].update(pg_clipfrac=result[1].detach().item(),
                                ppo_kl=result[2].detach().item(),
                                pg_clipfrac_lower=result[3].detach().item())
        self.pending = (["weighted_entropy"] if self.entropy_coeff != 0 else [])
        if self.use_kl_loss:
            self.pending.append("weighted_kl")
        return result

    def agg_loss(self, *args, **kwargs):
        if not self.pending:
            raise RuntimeError("Unexpected actor agg_loss call outside the pinned PG/H/KL call order.")
        component = self.pending.pop(0)
        result = self.aggregate(*args, **kwargs)
        coefficient = -self.entropy_coeff if component == "weighted_entropy" else self.kl_coef
        return self._scalar(result, component, coefficient)

    @contextmanager
    def installed(self):
        try:
            with _temporary_attribute(self.owner, "compute_policy_loss", self.compute_policy_loss):
                with _temporary_attribute(self.owner, "agg_loss", self.agg_loss):
                    yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


def _grpo_snapshot(data):
    # Official select deep-copies non_tensor_batch/meta_info, but not the tensor
    # container. The official TensorDict clone makes a separate container while
    # preserving read-only storages; rebinding advantages cannot change data.
    snapshot = data.select(deepcopy=True)
    snapshot.batch = data.batch.clone(recurse=False)
    snapshot.batch["advantages"] = data.batch["diagnostic_grpo_advantages"]
    return snapshot


def observe_native_optimizer_minibatch(actor, data, *, output_path, rank,
                                      policy_advantages=None):
    """Write complete minibatch statistics; return the last original metrics.

    Invoke collectively on the original FSDP ranks, with a separate output_path
    per rank. The driver creates GRPO advantages through the original global UID
    compute_advantage, then distributes the unchanged rows via the native worker
    dispatch. No reward, DT computation or advantage normalization occurs here.
    """
    import torch
    import numpy as np

    actor_owner = importlib.import_module(type(actor).__module__)
    core = importlib.import_module("verl.trainer.ppo.core_algos")
    original_update = getattr(type(actor), "update_policy").__get__(actor, type(actor))
    original_step = actor._optimizer_step
    actor_identity = _source_identity(type(actor))
    core_identity = _source_identity(core.compute_policy_loss)
    if actor_identity["sha256"] != ACTOR_SHA256 or core_identity["sha256"] != CORE_SHA256:
        raise ValueError("The requested scalar seams are for the recorded actual actor/core sources.")
    if actor_owner.compute_policy_loss is not core.compute_policy_loss or actor_owner.agg_loss is not core.agg_loss:
        raise ValueError("The actual actor loss aliases must still be the original core functions.")

    config = actor.config
    mini_rows = config.ppo_mini_batch_size
    micro_rows = config.ppo_micro_batch_size_per_gpu
    if len(data) != mini_rows or mini_rows != 32 or micro_rows != 4 or config.ppo_epochs != 1:
        raise ValueError("Supply one complete actual local32/B4/one-epoch optimizer minibatch.")
    if config.use_dynamic_bsz or config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("This observation targets the recorded static B4 vanilla owner path.")
    if config.entropy_coeff == 0 or not config.use_kl_loss:
        raise ValueError("The requested four observations require the original nonzero entropy and KL branches.")
    required = ("responses", "input_ids", "attention_mask", "position_ids", "old_log_probs",
                "ref_log_prob", "advantages", "diagnostic_grpo_advantages")
    if data.meta_info.get("multi_turn", False):
        required += ("loss_mask",)
    if policy_advantages is not None:
        required = tuple(key for key in required if key != "diagnostic_grpo_advantages")
    for key in required:
        if key not in data.batch:
            raise ValueError(f"Missing original minibatch field: {key}")
    if policy_advantages is None and data.batch["diagnostic_grpo_advantages"].shape != data.batch["advantages"].shape:
        raise ValueError("Externally computed official GRPO advantages must bind the same complete rows/tokens.")
    temperature = data.meta_info["temperature"]
    model = actor.actor_module
    parameters = dict(model.named_parameters())
    trainable = {name: p for name, p in parameters.items() if p.requires_grad}
    saved_grads = {name: (p.grad, None if p.grad is None else p.grad.detach().clone())
                   for name, p in parameters.items()}
    modes = [(module, module.training) for module in model.modules()]
    accumulation_existed = "gradient_accumulation" in vars(actor)
    saved_accumulation = vars(actor).get("gradient_accumulation")
    cuda_devices = sorted({tensor.device.index for tensor in data.batch.values()
                           if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"})
    snapshots, parameter_records, groups, pass_records = {}, {}, {}, {}
    optimizer = actor.actor_optimizer
    last_metrics = None
    if policy_advantages is None:
        labels = LABELS
        pass_data = {label: _grpo_snapshot(data) if label == "grpo_pg" else data
                     for label in labels}
    else:
        # Diagnostic coefficient views only. The original update_policy still
        # owns every loss, denominator, B4 accumulation and gradient reduction.
        labels = tuple(policy_advantages)
        pass_data = {}
        for label, advantages in policy_advantages.items():
            if advantages.shape != data.batch["advantages"].shape:
                raise ValueError("Diagnostic coefficient views must retain complete original rows/tokens.")
            snapshot = data.select(deepcopy=True)
            snapshot.batch = data.batch.clone(recurse=False)
            snapshot.batch["advantages"] = advantages
            pass_data[label] = snapshot
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            cpu_rng = torch.get_rng_state()
            cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
            for label in labels:
                pass_started = time.monotonic()
                print(f"native_minibatch_diagnostic rank={rank} label={label} phase=pass_start", flush=True)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(cpu_rng)
                for device, state in cuda_rng.items():
                    torch.cuda.set_rng_state(state, device)
                hooks = NativeLossHooks(actor_owner, label=label, entropy_coeff=config.entropy_coeff,
                                        use_kl_loss=config.use_kl_loss, kl_coef=config.kl_loss_coef,
                                        policy_component=policy_advantages is not None)
                step_records = []
                no_op_calls = []

                def no_op_optimizer_step(*args, **kwargs):
                    no_op_calls.append(True)
                    return None

                def observed_optimizer_step():
                    print(f"native_minibatch_diagnostic rank={rank} label={label} phase=optimizer_boundary elapsed={time.monotonic() - pass_started:.3f}s", flush=True)
                    if hooks.pending:
                        raise RuntimeError("Original scalar loss order did not finish before optimizer boundary.")
                    if step_records:
                        raise RuntimeError("This input executed more than one original optimizer minibatch.")
                    current, records = {}, {}
                    for name, parameter in trainable.items():
                        copied, record, bucket = capture_native_gradient(parameter, groups)
                        current[name] = (copied, bucket)
                        records[name] = record
                    snapshots[label] = current
                    parameter_records[label] = records
                    # The native clip and nonfinite handling execute unchanged.
                    norm = original_step()
                    step_records.append({"native_clip_return": norm.detach().item(),
                                         "native_clip_return_dtype": str(norm.dtype),
                                         "optimizer_step_no_op_calls": len(no_op_calls)})
                    print(f"native_minibatch_diagnostic rank={rank} label={label} phase=optimizer_boundary_complete elapsed={time.monotonic() - pass_started:.3f}s", flush=True)
                    return norm

                with hooks.installed():
                    with _temporary_attribute(optimizer, "step", no_op_optimizer_step):
                        with _temporary_attribute(actor, "_optimizer_step", observed_optimizer_step):
                            last_metrics = original_update(data=pass_data[label])
                if len(step_records) != 1:
                    raise RuntimeError("The complete original minibatch did not reach its optimizer boundary.")
                pass_records[label] = {"microbatch_losses": hooks.records,
                                       "optimizer_boundary": step_records[0], "original_metrics": last_metrics,
                                       "elapsed_seconds": time.monotonic() - pass_started}
                print(f"native_minibatch_diagnostic rank={rank} label={label} phase=pass_complete elapsed={time.monotonic() - pass_started:.3f}s", flush=True)
            statistics = native_gradient_statistics(
                trainable, snapshots, groups, data.batch["input_ids"].device, labels,
                norm_scope="Pre-clip complete local32 minibatch gradients from native backward; sharded DTensor scalar statistics SUM on its original mesh; replicated/plain contributions remain local.")
            result = {
                "scope": "One checkpoint-local complete original global64/local32 optimizer minibatch; diagnostic passes, no parameter or optimizer-state update.",
                "rank": rank, "labels": list(labels), "optimizer_step_executed": False,
                "native_gradient_clipping_executed": True, "scheduler_step_control": "Caller must no-op and restore the original worker scheduler step.",
                "sources": {"actor": actor_identity, "core": core_identity,
                            "diagnostic": _source_identity(observe_native_optimizer_minibatch),
                            "gradient_statistics": _source_identity(native_gradient_statistics)},
                "effective_config": {"ppo_mini_batch_size": mini_rows, "ppo_micro_batch_size_per_gpu": micro_rows,
                                     "ppo_epochs": config.ppo_epochs, "gradient_accumulation": mini_rows // micro_rows,
                                     "native_backward_scale": 1.0 / (mini_rows // micro_rows),
                                     "temperature": temperature, "multi_turn": data.meta_info.get("multi_turn", False),
                                     "entropy_coeff": config.entropy_coeff, "use_kl_loss": config.use_kl_loss,
                                     "kl_loss_coef": config.kl_loss_coef, "kl_loss_type": config.kl_loss_type,
                                     "loss_agg_mode": config.loss_agg_mode, "grad_clip": config.grad_clip},
                "input_fields": {key: {"shape": list(data.batch[key].shape), "dtype": str(data.batch[key].dtype)}
                                 for key in required},
                "coefficient_provenance": "Externally supplied diagnostic views of original saved coefficients; no normalization here." if policy_advantages is not None else "Provided by driver through official global-UID compute_advantage; not recomputed here.",
                "suppressed_backward_scope": "Scalar hooks return zero upstream gradients for other original branches; their forward/backward bodies still execute, including nonfinite intermediates.",
                "passes": pass_records, "parameters": parameter_records, "gradient_statistics": statistics,
            }
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        # Native zero_grad/clip must not consume or alter the caller's gradients.
        with torch.no_grad():
            for name, parameter in parameters.items():
                reference, copied = saved_grads[name]
                if reference is not None:
                    reference.copy_(copied)
                parameter.grad = reference
        for module, mode in modes:
            module.training = mode
        if accumulation_existed:
            actor.gradient_accumulation = saved_accumulation
        elif "gradient_accumulation" in vars(actor):
            del actor.gradient_accumulation
    return last_metrics
