"""CPU analysis of saved original three-branch VERL Adam observations.

Run only in the existing native Torch/VERL CPU environment with CUDA hidden.
No model, forward, rollout, backward, update, gather or Adam formula is used.
Original DataProto restores/chunks and core.agg_loss retain the actor B4 mask
and denominator. Parameter geometry describes saved local shards per rank.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

BRANCHES = ("dt", "grpo", "regularizers_only")
PROTO_SHA = "2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa"
CORE_SHA = "fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299"


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def source(function):
    function = inspect.unwrap(function)
    result = identity(inspect.getsourcefile(function))
    lines, first = inspect.getsourcelines(function)
    result.update(source_line=first, source_end_line=first + len(lines) - 1)
    return result


def tensor_identity(tensor, torch):
    array = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(array)).hexdigest()
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "logical_contiguous_native_dtype_bytes_sha256": digest,
            "elements": tensor.numel(), "bytes": tensor.numel() * tensor.element_size()}


def finite_counts(tensor, torch):
    return {"elements": tensor.numel(), "finite": int(torch.isfinite(tensor).sum()),
            "nan": int(torch.isnan(tensor).sum()),
            "positive_infinity": int(torch.isposinf(tensor).sum()),
            "negative_infinity": int(torch.isneginf(tensor).sum())}


def max_abs_or_none(tensor):
    return tensor.abs().max().item() if tensor.numel() else None


def resources(torch):
    import psutil
    process = psutil.Process()
    memory = process.memory_full_info()
    return {"pid": process.pid, "pid_birth": process.create_time(),
            "rss_bytes": memory.rss, "pss_bytes": getattr(memory, "pss", None),
            "host_available_bytes": psutil.virtual_memory().available,
            "cuda_initialized": torch.cuda.is_initialized(),
            "distributed_initialized": torch.distributed.is_initialized(),
            "torch_threads": torch.get_num_threads()}


def json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(item) for item in value]
    return value


def owner_observation(saved, before, after, agg_loss, loss_mode, torch):
    """Call the original reduction on continuous native local32/B4 chunks."""
    saved_chunks, before_chunks, after_chunks = saved.chunk(2), before.chunk(2), after.chunk(2)
    result = []
    for rank, (original, old, new) in enumerate(zip(saved_chunks, before_chunks, after_chunks)):
        records = []
        original_micro = original.batch.split(4)
        before_micro, after_micro = old.batch.split(4), new.batch.split(4)
        if len(original) != 32 or len(original_micro) != 8:
            raise ValueError("The requested saved native partition must be local32/B4x8.")
        for index, (batch, old_batch, new_batch) in enumerate(zip(original_micro, before_micro, after_micro)):
            width = batch["responses"].shape[-1]
            mask = batch["loss_mask"][:, -width:]
            old_H, new_H = old_batch["entropys"], new_batch["entropys"]
            old_LP, new_LP = old_batch["old_log_probs"], new_batch["old_log_probs"]
            original_LP = batch["old_log_probs"]
            if not all(value.shape == mask.shape for value in (old_H, new_H, old_LP, new_LP, original_LP)):
                raise ValueError("Saved observation and original actor action columns differ.")

            def reduce(value):
                return agg_loss(loss_mat=value, loss_mask=mask, loss_agg_mode=loss_mode).item()

            H_delta, LP_delta = new_H - old_H, new_LP - old_LP
            before_vs_original = old_LP - original_LP
            selected = mask != 0
            records.append({
                "rank": rank, "microbatch_index": index,
                "global_actor_rows": list(range(rank * 32 + index * 4, rank * 32 + index * 4 + 4)),
                "traj_uid": [str(x) for x in original.non_tensor_batch["traj_uid"][index*4:index*4+4]],
                "policy_mask_denominator": mask.sum().item(),
                "H_before": reduce(old_H), "H_after": reduce(new_H), "H_delta": reduce(H_delta),
                "LP_before": reduce(old_LP), "LP_after": reduce(new_LP), "LP_delta": reduce(LP_delta),
                "LP_delta_abs_mean": reduce(LP_delta.abs()),
                "LP_delta_RMS": math.sqrt(reduce(LP_delta.square())),
                "LP_delta_max_abs_selected": max_abs_or_none(LP_delta[selected]),
                "before_LP_minus_saved_original_mean": reduce(before_vs_original),
                "before_LP_minus_saved_original_abs_mean": reduce(before_vs_original.abs()),
                "before_LP_minus_saved_original_max_abs_selected": max_abs_or_none(before_vs_original[selected]),
                "before_LP_equals_original_selected": torch.equal(old_LP[selected], original_LP[selected]),
                "finite_selected": {label: finite_counts(value[selected], torch)
                                    for label, value in (("H_before", old_H), ("H_after", new_H),
                                                         ("LP_before", old_LP), ("LP_after", new_LP))},
            })
        aggregate_keys = ("H_before", "H_after", "H_delta", "LP_before", "LP_after", "LP_delta",
                          "LP_delta_abs_mean", "before_LP_minus_saved_original_mean",
                          "before_LP_minus_saved_original_abs_mean")
        result.append({"rank": rank, "B4": records, "gradient_accumulation": 8,
                       "native_equal_B4_sum_divided_by_8": {
                           key: sum(row[key] for row in records) / 8 for key in aggregate_keys}})
    keys = result[0]["native_equal_B4_sum_divided_by_8"]
    return {"rank_reductions": result, "two_rank_equal_mean_of_native_reductions": {
        key: sum(row["native_equal_B4_sum_divided_by_8"][key] for row in result) / 2
        for key in keys}}


def local_parameter_geometry(rank, payloads, receipts, torch):
    """FP64 descriptive copies of native storage values; never join ranks."""
    names = sorted(payloads["dt"]["before"])
    for payload in payloads.values():
        if sorted(payload["before"]) != names or sorted(payload["after"]) != names:
            raise ValueError("Saved native branches do not expose identical parameter coordinates.")
    pairs = [(a, b) for i, a in enumerate(BRANCHES) for b in BRANCHES[i:]]
    gram = {f"{a}:{b}": 0.0 for a, b in pairs}
    difference_squared = {f"{a}:{b}": 0.0 for a, b in pairs if a != b}
    initial = {branch: {"exact_tensor_count": 0, "tensor_count": len(names),
                        "max_abs_diff_from_dt": 0.0, "squared_difference_from_dt": 0.0}
               for branch in BRANCHES}
    branch_summary = {branch: {"delta_elements": 0, "delta_nonzero_elements": 0,
                               "delta_nonfinite_elements": 0, "gradient_none_parameters": 0,
                               "gradient_nonfinite_elements": 0, "gradient_squared_norm": 0.0,
                               "dot_delta_and_negative_optimizer_input_gradient": 0.0,
                               "gradient_direction_available": receipts[branch]["optimizer_step_calls"] == 1}
                      for branch in BRANCHES}
    parameter_records = []
    for name in names:
        reference = payloads["dt"]["before"][name]
        deltas, record = {}, {"name": name, "branches": {}}
        for branch in BRANCHES:
            payload = payloads[branch]
            before, after = payload["before"][name], payload["after"][name]
            if before.shape != after.shape or before.shape != reference.shape:
                raise ValueError("Saved native parameter shapes differ: " + name)
            exact = before.dtype == reference.dtype and torch.equal(before, reference)
            initial[branch]["exact_tensor_count"] += int(exact)
            init_delta = before.double() - reference.double()
            init_max = max_abs_or_none(init_delta)
            if init_max is not None:
                initial[branch]["max_abs_diff_from_dt"] = max(initial[branch]["max_abs_diff_from_dt"], init_max)
            initial[branch]["squared_difference_from_dt"] += torch.dot(init_delta.reshape(-1), init_delta.reshape(-1)).item()
            delta = after.double() - before.double()
            deltas[branch] = delta.reshape(-1)
            stats = branch_summary[branch]
            stats["delta_elements"] += delta.numel()
            stats["delta_nonzero_elements"] += int(delta.count_nonzero())
            stats["delta_nonfinite_elements"] += int((~torch.isfinite(delta)).sum())
            tensor_record = {"before": tensor_identity(before, torch), "after": tensor_identity(after, torch),
                             "before_layout": payload["before_layout"][name],
                             "after_layout": payload["after_layout"][name],
                             "before_finite": finite_counts(before, torch),
                             "after_finite": finite_counts(after, torch),
                             "native_dtype_preserved": before.dtype == after.dtype,
                             "initial_equal_to_dt": exact}
            gradients = payload["optimizer_input_gradients_after_native_clip"]
            if stats["gradient_direction_available"] and len(gradients) == 1:
                gradient = next(iter(gradients.values()))[name]
                if gradient is None:
                    stats["gradient_none_parameters"] += 1
                    tensor_record["optimizer_input_gradient"] = None
                    tensor_record["gradient_layout"] = next(iter(payload["gradient_layout"].values()))[name]
                else:
                    tensor_record["optimizer_input_gradient"] = tensor_identity(gradient, torch)
                    tensor_record["gradient_finite"] = finite_counts(gradient, torch)
                    tensor_record["gradient_layout"] = next(iter(payload["gradient_layout"].values()))[name]
                    g = gradient.reshape(-1).double()
                    stats["gradient_nonfinite_elements"] += int((~torch.isfinite(g)).sum())
                    stats["gradient_squared_norm"] += torch.dot(g, g).item()
                    stats["dot_delta_and_negative_optimizer_input_gradient"] -= torch.dot(deltas[branch], g).item()
            else:
                stats["gradient_direction_available"] = False
                tensor_record["optimizer_input_gradient"] = "not a single saved native step"
            record["branches"][branch] = tensor_record
        for a, b in pairs:
            gram[f"{a}:{b}"] += torch.dot(deltas[a], deltas[b]).item()
            if a != b:
                difference = deltas[a] - deltas[b]
                difference_squared[f"{a}:{b}"] += torch.dot(difference, difference).item()
        parameter_records.append(record)
    norms = {branch: math.sqrt(gram[f"{branch}:{branch}"]) for branch in BRANCHES}
    cosines = {f"{a}:{b}": gram[f"{a}:{b}"]/(norms[a]*norms[b]) if norms[a]*norms[b] else None
               for a, b in pairs}
    for branch, stats in branch_summary.items():
        grad_norm = math.sqrt(stats["gradient_squared_norm"])
        stats["gradient_norm"] = grad_norm if stats["gradient_direction_available"] else None
        stats["cosine_delta_to_negative_optimizer_input_gradient"] = (
            stats["dot_delta_and_negative_optimizer_input_gradient"]/(norms[branch]*grad_norm)
            if stats["gradient_direction_available"] and norms[branch]*grad_norm else None)
    return {"rank": rank, "scope": "Local native rank shards only; no gather or cross-rank gradient/norm inference.",
            "arithmetic": "FP64 descriptive differences/dots of unchanged original-dtype saved before/after values.",
            "initial_parameter_comparison": initial, "parameter_delta_Gram": gram,
            "parameter_delta_norms": norms, "parameter_delta_cosines": cosines,
            "pairwise_parameter_delta_difference_norms": {key: math.sqrt(value) for key, value in difference_squared.items()},
            "branches": branch_summary, "parameters": parameter_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--minibatch-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run only in the existing CPU environment with CUDA_VISIBLE_DEVICES=''.")
    import torch
    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import agg_loss
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("This analysis must not initialize CUDA or a process group.")
    started, before_resources = time.time(), resources(torch)
    sources = {"analyzer": identity(__file__), "protocol_load": source(DataProto.load_from_disk),
               "protocol_chunk": source(DataProto.chunk), "agg_loss": source(agg_loss),
               "torch": identity(torch.__file__)}
    if sources["protocol_load"]["sha256"] != PROTO_SHA or sources["protocol_chunk"]["sha256"] != PROTO_SHA or sources["agg_loss"]["sha256"] != CORE_SHA:
        raise ValueError("Use the original recorded protocol and loss reduction owner.")
    completed_path = args.input_dir / "completed.json"
    inspection_path = args.input_dir / "native-owner-inspection.json"
    completed = json.loads(completed_path.read_text())
    inspection = json.loads(inspection_path.read_text())
    minibatch = identity(args.minibatch_path)
    if minibatch["sha256"] != completed["source_minibatch_sha256"]:
        raise ValueError("The input is not the saved original native minibatch.")
    saved = DataProto.load_from_disk(str(args.minibatch_path))
    if len(saved) != 64:
        raise ValueError("Use the recorded original global64 snapshot.")
    mode = inspection["actor_config"]["actor"]["loss_agg_mode"]
    input_sources = {"minibatch": minibatch, "completed": identity(completed_path),
                     "inspection": identity(inspection_path)}
    observations, update_receipts = {}, {rank: {} for rank in range(2)}
    before_arrays = {}
    completed_branches = {record["branch"]: record for record in completed["branches"]}
    for branch in BRANCHES:
        paths = [args.input_dir / f"{branch}-{when}-logprob.pkl" for when in ("before", "after")]
        for when, path in zip(("before", "after"), paths):
            input_sources[path.name] = identity(path)
            if input_sources[path.name]["sha256"] != completed_branches[branch][when]["sha256"]:
                raise ValueError("Saved native observation artifact SHA differs from its completion receipt.")
        old, new = [DataProto.load_from_disk(str(path)) for path in paths]
        if len(old) != 64 or len(new) != 64:
            raise ValueError("Native observation row count differs from saved64.")
        observations[branch] = owner_observation(saved, old, new, agg_loss, mode, torch)
        before_arrays[branch] = old.batch
        for rank in range(2):
            receipt_path = args.input_dir / f"rank{rank}-{branch}-native-adam.json"
            input_sources[receipt_path.name] = identity(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            update_receipts[rank][branch] = receipt
    initial_readout_comparisons = {}
    for branch in BRANCHES:
        initial_readout_comparisons[branch] = {}
        for field in ("entropys", "old_log_probs"):
            a, b = before_arrays[branch][field], before_arrays["dt"][field]
            initial_readout_comparisons[branch][field] = {
                "exact": a.dtype == b.dtype and torch.equal(a, b),
                "max_abs_storage_value_diff_from_dt": max_abs_or_none(a.double()-b.double()),
                "tensor": tensor_identity(a, torch)}
    geometry, native_updates = [], {}
    for rank in range(2):
        payloads = {}
        native_updates[str(rank)] = {}
        for branch in BRANCHES:
            receipt = update_receipts[rank][branch]
            path = args.input_dir / f"rank{rank}-{branch}-native-parameter-shards.pt"
            input_sources[path.name] = identity(path)
            if input_sources[path.name]["sha256"] != receipt["parameter_shards"]["sha256"]:
                raise ValueError("Actual saved native parameter shard artifact SHA differs.")
            payloads[branch] = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            native_updates[str(rank)][branch] = {key: receipt[key] for key in (
                "status", "optimizer_step_calls", "scheduler_step_calls", "optimizer_before",
                "optimizer_after", "scheduler_before", "scheduler_after", "original_metrics",
                "optimizer_steps", "scheduler_steps", "effective_actor_config",
                "sources", "resources_before", "resources_after")}
        geometry.append(local_parameter_geometry(rank, payloads, update_receipts[rank], torch))
        del payloads
    result = {
        "scope": "Saved native single actual AdamW updates only, three branches at the same complete checkpoint25. CPU analysis; no model or new execution. Not historical multi-step causal proof.",
        "sources": sources, "inputs": input_sources,
        "owner_reduction": {"loss_agg_mode": mode, "partition": "Original DataProto.chunk(2), continuous local32 -> 8 B4s -> sum/8; equal mean of two rank reductions separately reported.",
                            "mask": "Saved original loss_mask[:, -responses_width:], identical for H/LP before/after. No global pooled mean replaces the native denominator.",
                            "scalar_precision": "Each B4 uses original agg_loss on original observation dtype. Python arithmetic combines its original scalar outputs; FP64 copies only describe parameter/readout storage differences."},
        "H_LP_observations": observations, "initial_readout_comparisons": initial_readout_comparisons,
        "actual_native_updates": native_updates, "local_parameter_geometry": geometry,
        "interpretation_limits": [
            "regularizers_only means current advantages are zero. The restored AdamW moments and weight decay still contain the common prior history; its parameter delta is not a pure H/KL vector.",
            "Differences between actual branch parameter deltas are matched experimental differences, not a linear PG/H/KL decomposition of AdamW or a percentage attribution.",
            "Entropy/LP values are original before/after compute_log_prob observations on the same fixed saved batch. Their changes are not new rollout success rates or a complete policy distribution divergence.",
            "Parameter Gram/cosines describe local native shards; no cross-rank gather/global gradient tolerance or new numerical pass threshold is introduced.",
        ],
        "operations": {"model_initializations":0,"forwards":0,"sampling":0,"backward":0,
                       "optimizer_updates":0,"gathers":0,"credit_recomputations":0},
        "runtime": {"started_unix":started,"completed_unix":time.time(),
                    "resources_before":before_resources,"resources_after":resources(torch)},
        "nonfinite_JSON_encoding": "Nonfinite scalar descriptions are explicit NaN/+Infinity/-Infinity strings; tensor finite counts are retained.",
        "empty_storage_description": "Empty local shards have zero dot/difference contribution. A max over empty selected storage is unavailable (null), not a numerical pass result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_finite(result), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output":identity(args.output),"cuda_initialized":torch.cuda.is_initialized(),
                      "distributed_initialized":torch.distributed.is_initialized()}))


if __name__ == "__main__":
    main()
