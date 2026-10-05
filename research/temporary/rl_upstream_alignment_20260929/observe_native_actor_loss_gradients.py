"""Observe original actor loss gradients on an already prepared response B4.

This diagnostic does not construct a model or PPO batch, change coefficients,
or execute an optimizer step. Event/readout queries must not enter this batch.
The caller owns loading the real checkpoint and the worker's normal device and
sharding context. This is a checkpoint-local response observation, not recovery
of a historical full trajectory or optimizer minibatch.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time


def _source_identity(value):
    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        return {"path": None, "sha256": None}
    path = Path(source).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def capture_native_gradient(parameter, groups):
    """Shared diagnostic snapshot; preserve native backward/DTensor ownership."""
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard

    grad = parameter.grad
    tensor = parameter if grad is None else grad
    record = {"shape": list(parameter.shape), "parameter_dtype": str(parameter.dtype),
              "grad_dtype": None if grad is None else str(grad.dtype), "grad_is_none": grad is None}
    bucket = ("local_plain",)
    if isinstance(tensor, DTensor):
        mesh = tensor.device_mesh
        placements = tensor.placements
        record.update(mesh=mesh.mesh.detach().cpu().tolist(), mesh_device_type=mesh.device_type,
                      placements=[str(p) for p in placements])
        if mesh.ndim != 1:
            raise ValueError("Actual gradient mesh is outside this observation's recorded one-dimensional owner path.")
        mesh_key = (mesh.device_type, tuple(mesh.mesh.detach().cpu().reshape(-1).tolist()))
        if isinstance(placements[0], Shard):
            bucket = ("sharded",) + mesh_key
            groups[bucket] = mesh.get_group(0)
        elif isinstance(placements[0], Replicate):
            bucket = ("replicated",) + mesh_key
        else:
            raise ValueError(f"Cannot label gradient statistics for actual placement {placements} without owner materialization.")
        if grad is not None:
            grad = grad.detach().to_local()
            from torch.distributed._functional_collectives import AsyncCollectiveTensor
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
    record["statistics_bucket"] = list(bucket)
    copied = None if grad is None else grad.detach().to(device="cpu", copy=True)
    if copied is not None:
        record["local_grad_shape"] = list(copied.shape)
    return copied, record, bucket


def native_gradient_statistics(trainable, snapshots, groups, device, labels, *, norm_scope):
    """Shared FP64 scalar statistics; never replace the native gradient reducer."""
    import torch

    pair_keys = tuple((a, b) for i, a in enumerate(labels) for b in labels[i:])
    buckets = {}
    for name in trainable:
        for i, (a, b) in enumerate(pair_keys):
            ga, bucket_a = snapshots[a][name]
            gb, bucket_b = snapshots[b][name]
            if bucket_a != bucket_b:
                raise ValueError(f"Native gradient ownership changed between fresh forwards: {name}")
            values = buckets.setdefault(bucket_a, [0.0] * len(pair_keys))
            if ga is not None and gb is not None:
                values[i] += torch.dot(ga.reshape(-1).double(), gb.reshape(-1).double()).item()
    totals = [0.0] * len(pair_keys)
    for bucket, values in buckets.items():
        if bucket[0] == "sharded":
            group = groups[bucket]
            statistic_device = device if torch.distributed.get_backend(group) == "nccl" else "cpu"
            values_tensor = torch.tensor(values, dtype=torch.float64, device=statistic_device)
            torch.distributed.all_reduce(values_tensor, op=torch.distributed.ReduceOp.SUM, group=group)
            values = values_tensor.cpu().tolist()
            del values_tensor
        totals = [total + value for total, value in zip(totals, values)]
    return {
        "measurement_dtype": "torch.float64",
        "ownership_buckets": [list(bucket) for bucket in buckets],
        "norm_scope": norm_scope,
        "inner_products": {f"{a}:{b}": value for (a, b), value in zip(pair_keys, totals)},
        "norms": {a: value ** 0.5 for (a, b), value in zip(pair_keys, totals) if a == b},
    }


def observe_native_actor_loss_gradients(worker, micro_batch, *, temperature, multi_turn=True):
    """Return PG, weighted entropy and weighted KL gradient norms/inner products.

    ``micro_batch`` is the original-ID PPO representation of four actual current
    responses, including their real preceding history. It must supply responses,
    input_ids, attention_mask, position_ids, advantages, and (for multi_turn)
    loss_mask. The source advantage is supplied by the original DT combination.
    Old/ref log-probs are freshly computed through the original actor at this
    same checkpoint. Missing past/future action masks are not reconstructed.

    Invoke collectively on all FSDP ranks in the normal loaded worker context.
    Original backward synchronizes gradients. Sharded gradient *statistics* are
    summed through the DTensor's own mesh group; gradient tensors are not moved
    through a replacement reducer or materialized by a CPU/NCCL full gather.
    Plain parameter results are explicitly local, without an inferred DP sum.
    """
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard

    actor = worker.actor
    model = actor.actor_module
    config = actor.config
    core = importlib.import_module("verl.trainer.ppo.core_algos")
    actor_owner = importlib.import_module(type(actor).__module__)
    # Use the same imported function objects as the actual actor update.
    policy_loss = actor_owner.compute_policy_loss
    agg_loss = actor_owner.agg_loss
    kl_penalty = actor_owner.kl_penalty
    if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("This observation was requested for the actual vanilla PPO owner path.")
    if not worker._is_lora or not config.use_kl_loss:
        raise ValueError("The requested reference path is the actual LoRA disable_adapter KL path.")

    responses = micro_batch["responses"]
    batch_rows, response_length = responses.shape
    if batch_rows != 4:
        raise ValueError("Supply the four actual current responses as the original B4 PPO batch.")
    attention_tail = micro_batch["attention_mask"][:, -response_length:]
    response_mask = micro_batch["loss_mask"][:, -response_length:] if multi_turn else attention_tail
    advantages = micro_batch["advantages"]
    if advantages.shape != responses.shape or response_mask.shape != responses.shape:
        raise ValueError("Current response IDs, original advantages and actor mask must align.")

    clip = config.clip_ratio
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip
    mini_rows = config.ppo_mini_batch_size
    micro_rows = config.ppo_micro_batch_size_per_gpu
    accumulation = mini_rows // micro_rows
    backward_scale = batch_rows / mini_rows if config.use_dynamic_bsz else 1.0 / accumulation
    labels = ("pg", "weighted_entropy", "weighted_kl")
    pair_keys = tuple((a, b) for i, a in enumerate(labels) for b in labels[i:])
    named_parameters = dict(model.named_parameters())
    trainable = {name: parameter for name, parameter in named_parameters.items() if parameter.requires_grad}
    saved_grads = {name: parameter.grad for name, parameter in named_parameters.items()}
    modes = [(module, module.training) for module in model.modules()]
    cuda_devices = sorted({value.device.index for value in micro_batch.values()
                           if isinstance(value, torch.Tensor) and value.device.type == "cuda"})
    snapshots = {}
    parameter_records = {}
    groups = {}
    forwards = {}
    result = {
        "scope": "Same checkpoint, real current-response B4 at its recomputed old-policy point; not a historical complete update replay.",
        "original_backward_reduction_scope": "Local B4 per worker; native FSDP backward performs its original cross-rank gradient reduction. This is a current response subset contribution, not the complete optimizer minibatch.",
        "worker_world_size": worker.world_size,
        "optimizer_step_executed": False,
        "gradient_clipping_executed": False,
        "sources": {"actor_forward": _source_identity(actor._forward_micro_batch),
                    "compute_policy_loss": _source_identity(policy_loss),
                    "agg_loss": _source_identity(agg_loss), "kl_penalty": _source_identity(kl_penalty),
                    "core_module": _source_identity(core.compute_policy_loss)},
        "actor_loss_imports_match_core": {"compute_policy_loss": policy_loss is core.compute_policy_loss,
                                           "agg_loss": agg_loss is core.agg_loss,
                                           "kl_penalty": kl_penalty is core.kl_penalty},
        "effective_config": {"ppo_mini_batch_size": mini_rows,
                             "ppo_micro_batch_size_per_gpu": micro_rows,
                             "use_dynamic_bsz": config.use_dynamic_bsz,
                             "gradient_accumulation": accumulation,
                             "backward_scale": backward_scale, "temperature": temperature,
                             "clip_ratio": clip, "clip_ratio_low": clip_low, "clip_ratio_high": clip_high,
                             "clip_ratio_c": config.get("clip_ratio_c", 3.0),
                             "entropy_coeff": config.entropy_coeff, "kl_loss_coef": config.kl_loss_coef,
                             "kl_loss_type": config.kl_loss_type, "loss_agg_mode": config.loss_agg_mode},
        "mask": {"multi_turn": multi_turn, "actor_mask_tokens": response_mask.sum().item(),
                 "attention_tail_tokens": attention_tail.sum().item(),
                 "actor_mask_equals_attention_tail": torch.equal(response_mask, attention_tail)},
        "input_dtypes": {name: str(value.dtype) for name, value in micro_batch.items()
                         if isinstance(value, torch.Tensor)},
        "components": {},
    }

    def scalar(value):
        return value.detach().item()

    try:
        # Preserve caller RNG while giving the three fresh train forwards the
        # same RNG start. Native activation-checkpoint RNG behavior is unchanged.
        with torch.random.fork_rng(devices=cuda_devices):
            cpu_rng = torch.get_rng_state()
            cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}

            def reset_rng():
                torch.set_rng_state(cpu_rng)
                for device, state in cuda_rng.items():
                    torch.cuda.set_rng_state(state, device)

            model.eval()
            with torch.no_grad():
                reset_rng()
                _, old_log_prob = actor._forward_micro_batch(micro_batch, temperature=temperature,
                                                            calculate_entropy=True)
                old_log_prob = old_log_prob.detach()
                reset_rng()
                with model.disable_adapter():
                    _, ref_log_prob = actor._forward_micro_batch(micro_batch, temperature=temperature,
                                                                calculate_entropy=True)
                ref_log_prob = ref_log_prob.detach()
            result["recomputed_log_prob_dtypes"] = {"old": str(old_log_prob.dtype), "ref": str(ref_log_prob.dtype)}

            for label in labels:
                model.zero_grad(set_to_none=True)
                model.train()
                reset_rng()
                with torch.enable_grad():
                    entropy, log_prob = actor._forward_micro_batch(micro_batch, temperature=temperature,
                                                                 calculate_entropy=config.entropy_coeff != 0)
                    pg, clipfrac, ppo_kl, lower_clipfrac = policy_loss(
                        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
                        response_mask=response_mask, cliprange=clip, cliprange_low=clip_low,
                        cliprange_high=clip_high, clip_ratio_c=config.get("clip_ratio_c", 3.0),
                        loss_agg_mode=config.loss_agg_mode)
                    weighted_entropy = -config.entropy_coeff * agg_loss(
                        loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)
                    weighted_kl = config.kl_loss_coef * agg_loss(
                        loss_mat=kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                                           kl_penalty=config.kl_loss_type),
                        loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)
                    owner_terms = {"pg": pg, "weighted_entropy": weighted_entropy, "weighted_kl": weighted_kl}
                    observed_loss = owner_terms[label] * backward_scale
                    selected_ratio = torch.exp(log_prob.detach() - old_log_prob)[response_mask.bool()]
                    result["components"][label] = {
                        "owner_loss_unscaled": scalar(owner_terms[label]),
                        "loss_passed_to_backward": scalar(observed_loss),
                        "all_owner_terms_unscaled": {key: scalar(value) for key, value in owner_terms.items()},
                        "log_prob_dtype": str(log_prob.dtype), "entropy_dtype": str(entropy.dtype),
                        "loss_dtype": str(observed_loss.dtype),
                        "pg_clipfrac": scalar(clipfrac), "ppo_kl": scalar(ppo_kl),
                        "pg_clipfrac_lower": scalar(lower_clipfrac),
                        "ratio": {"dtype": str(selected_ratio.dtype), "tokens": selected_ratio.numel(),
                                  "min": scalar(selected_ratio.min()) if selected_ratio.numel() else None,
                                  "max": scalar(selected_ratio.max()) if selected_ratio.numel() else None,
                                  "mean": scalar(selected_ratio.mean()) if selected_ratio.numel() else None},
                    }
                    forwards[label] = log_prob.detach().to(device="cpu", copy=True)
                    observed_loss.backward()
                    del observed_loss, owner_terms, pg, weighted_entropy, weighted_kl
                    del entropy, log_prob, clipfrac, ppo_kl, lower_clipfrac, selected_ratio
                snapshots[label] = {}
                parameter_records[label] = {}
                for name, parameter in trainable.items():
                    copied, record, bucket = capture_native_gradient(parameter, groups)
                    snapshots[label][name] = (copied, bucket)
                    parameter_records[label][name] = record
                model.zero_grad(set_to_none=True)

            # Statistics are measured in FP64 on independent native-dtype
            # snapshots; no training tensor is recast or modified.
            result["gradient_statistics"] = native_gradient_statistics(
                trainable, snapshots, groups, micro_batch["input_ids"].device, labels,
                norm_scope="Logical DTensor gradients after native backward; plain tensors remain local. No clipping or optimizer step.")
            result["fresh_forward_log_prob_equal"] = {f"{a}:{b}": torch.equal(forwards[a], forwards[b])
                                                      for a, b in pair_keys if a != b}
            result["parameters"] = parameter_records
            result["trainable_parameter_count"] = len(trainable)
            result["trainable_non_lora_names"] = [name for name in trainable if "lora_" not in name]
    finally:
        # Restore pre-observation state, including any original gradient objects.
        model.zero_grad(set_to_none=True)
        for name, parameter in named_parameters.items():
            parameter.grad = saved_grads[name]
        for module, training in modes:
            module.training = training
    return result


def observe_gradients(worker, samples, out, save):
    """Thin original-ID/current-response adapter for the four recorded samples.

    History IDs condition each response. Historical policy/observation routing
    outside that response is not reconstructed or credited in this diagnostic.
    Original VERL padding/positions and original DT Q/V/A composition are reused.
    """
    import torch
    from counterfactual import reward_event_token_credit
    from verl.utils.device import get_torch_device
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_2d_list_to_length, pad_sequence_to_length

    if len(samples) != 4:
        raise ValueError("The observation requires the four recorded current responses.")
    prefixes, response_ids, signed, rewards = [], [], [], []
    sample_metadata = []
    for sample in samples:
        ids = sample["selected_input_ids"]
        start, end = sample["source_start"], sample["source_end"]
        values = sample["source_signed"]
        if not 0 < start < end <= len(ids) or len(values) != end - start:
            raise ValueError("The original source range and original signed vector do not align.")
        prefixes.append(torch.tensor(ids[:start], dtype=torch.long).unsqueeze(0))
        response_ids.append(ids[start:end])
        # Actual stopped EventRatioReadout copies raw FP64 signed entries to
        # its FP32 row vector before the unchanged Q/V/A composition.
        signed.append(torch.tensor(values, dtype=torch.float32).unsqueeze(0))
        rewards.append([sample["observed_return"]])
        sample_metadata.append({"traj_uid": sample["traj_uid"], "source_start": start,
                                "source_end": end, "prefix_tokens": start,
                                "current_response_tokens": end-start,
                                "observed_return": sample["observed_return"]})
    pad = worker.tokenizer.pad_token_id
    prefix_width = max(value.shape[-1] for value in prefixes)
    prompts = torch.cat([pad_sequence_to_length(value, prefix_width, pad, left_pad=True)
                         for value in prefixes], dim=0)
    prefix_mask = torch.cat([pad_sequence_to_length(torch.ones_like(value), prefix_width, 0, left_pad=True)
                            for value in prefixes], dim=0)
    responses = pad_2d_list_to_length(response_ids, pad)
    action_mask = pad_2d_list_to_length([[1] * len(ids) for ids in response_ids], 0).bool()
    original_signed = torch.cat([pad_sequence_to_length(value, responses.shape[-1], 0)
                                 for value in signed], dim=0)
    credit = reward_event_token_credit(
        original_signed.unsqueeze(1), torch.tensor(rewards, dtype=torch.float32),
        action_mask.unsqueeze(1), action_mask)
    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prefix_mask, action_mask.long()), dim=-1)
    batch = {"input_ids": input_ids, "responses": responses, "attention_mask": attention_mask,
             "position_ids": compute_position_id_with_mask(attention_mask),
             "loss_mask": torch.cat((torch.zeros_like(prefix_mask), action_mask.long()), dim=-1),
             # Match the original DT-to-actor tensor boundary after composition.
             "advantages": credit.advantages.to(dtype=torch.float32)}
    device = get_torch_device().current_device()
    batch = {key: value.to(device) for key, value in batch.items()}
    out = Path(out)
    path = out / f"rank{worker.rank}-gradients.json"
    started = time.time()
    save("native_actor_gradient_observation_started")
    try:
        if worker._is_offload_param:
            load_fsdp_model_to_gpu(worker.actor_module_fsdp)
        result = observe_native_actor_loss_gradients(
            worker, batch, temperature=worker.config.rollout.temperature, multi_turn=True)
        result.update(rank=worker.rank, observed_unix=time.time(), seconds=time.time()-started,
                      samples=sample_metadata,
                      current_response_mask_provenance="Only the original source_start:end policy-action span; no past/future action mask reconstructed.",
                      batch_scope="Constructed current-response-only B4 from exact recorded IDs; not the original complete historical train batch.",
                      qva_composition_source=_source_identity(reward_event_token_credit),
                      row_signed_dtype=str(original_signed.dtype),
                      original_source_signed_storage='Original JSON values emitted from the owner FP64 contraction',
                      qva_composition_dtype=str(credit.advantages.dtype),
                      actor_advantages_dtype=str(batch["advantages"].dtype),
                      batch_owner_sources={"padding": _source_identity(pad_sequence_to_length),
                                           "list_padding": _source_identity(pad_2d_list_to_length),
                                           "positions": _source_identity(compute_position_id_with_mask)},
                      checkpoint_scope="Already loaded original worker checkpoint; no additional model initialization or optimizer step.")
        path.write_text(json.dumps(result, indent=2) + "\n")
        save("native_actor_gradient_observation_completed")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "result": result}
    finally:
        if worker._is_offload_param:
            offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
