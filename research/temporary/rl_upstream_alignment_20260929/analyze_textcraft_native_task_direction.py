"""CPU descriptions of saved native total gradients and actual Adam displacements.

Read the six existing rank/branch parameter payloads; never initialize a model,
call a loss, differentiate, update, gather, or simulate Adam. Total-gradient
differences are descriptive measurements, not an exact component identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


BRANCHES = ("dt", "grpo", "regularizers_only")
VECTORS = ("dt_total", "grpo_total", "regularizers_total",
           "dt_total_minus_regularizers_total", "grpo_total_minus_regularizers_total")
LAYOUT_KEYS = ("global_shape", "dtype", "placements", "mesh", "mesh_device_type",
               "local_shape", "local_elements", "local_bytes")


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def tensor_identity(value, torch):
    storage = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "elements": value.numel(), "bytes": value.numel() * value.element_size(),
            "logical_contiguous_native_dtype_bytes_sha256": hashlib.sha256(memoryview(storage)).hexdigest()}


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


def cosine(dot, norm_a, norm_b):
    denominator = norm_a * norm_b
    return dot / denominator if denominator else None


def json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(item) for item in value]
    return value


def read_json(path, inputs):
    inputs[Path(path).name] = identity(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def layout_equal(a, b):
    return all(a.get(key) == b.get(key) for key in LAYOUT_KEYS)


def rank_geometry(rank, payloads, receipts, torch):
    names = sorted(payloads["dt"]["before"])
    gradients, gradient_layouts = {}, {}
    for branch, payload in payloads.items():
        if payload["rank"] != rank or payload["branch"] != branch:
            raise ValueError("Saved payload rank/branch is not its recorded coordinate.")
        for field in ("before", "after", "before_layout", "after_layout"):
            if sorted(payload[field]) != names:
                raise ValueError("Saved parameter coordinates differ: " + field)
        steps = payload["optimizer_input_gradients_after_native_clip"]
        layouts = payload["gradient_layout"]
        if receipts[branch]["optimizer_step_calls"] != 1 or set(steps) != {"0"} or set(layouts) != {"0"}:
            raise ValueError("Use the already recorded single actual native optimizer step.")
        gradients[branch], gradient_layouts[branch] = steps["0"], layouts["0"]
        if sorted(gradients[branch]) != names or sorted(gradient_layouts[branch]) != names:
            raise ValueError("Saved gradient coordinates differ from native trainable parameters.")

    pairs = [(a, b) for i, a in enumerate(VECTORS) for b in VECTORS[i:]]
    gram = {f"{a}:{b}": 0.0 for a, b in pairs}
    delta_norm2 = {branch: 0.0 for branch in BRANCHES}
    common_delta_norm2 = {branch: 0.0 for branch in BRANCHES}
    dots = {vector: {branch: 0.0 for branch in BRANCHES} for vector in VECTORS}
    initial_counts = Counter()
    conditions = {"initial_native_dtype_parameters_exact_across_branches": True,
                  "parameter_layouts_identical_across_branches_and_before_after": True,
                  "gradient_layout_matches_parameter_and_all_branches": True,
                  "all_parameters_fp32": True, "all_non_none_gradients_fp32": True,
                  "all_saved_parameters_and_gradients_finite": True,
                  "gradient_none_pattern_identical_across_branches": True}
    none_counts, dtype_counts = {branch: 0 for branch in BRANCHES}, Counter()
    comparable_parameters, comparable_elements = 0, 0
    records = []
    for name in names:
        reference = payloads["dt"]["before"][name]
        deltas, native_gradients = {}, {}
        row = {"name": name, "branches": {}}
        for branch in BRANCHES:
            payload = payloads[branch]
            before, after, gradient = payload["before"][name], payload["after"][name], gradients[branch][name]
            if not all(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                       for tensor in (before, after)):
                raise TypeError("Expected original CPU parameter tensor storage.")
            if before.shape != after.shape or before.shape != reference.shape:
                raise ValueError("Native before/after parameter shapes differ: " + name)
            exact = before.dtype == reference.dtype and torch.equal(before, reference)
            initial_counts[branch] += int(exact)
            conditions["initial_native_dtype_parameters_exact_across_branches"] &= exact
            parameter_layout_ok = (layout_equal(payload["before_layout"][name], payload["after_layout"][name])
                                   and layout_equal(payload["before_layout"][name], payloads["dt"]["before_layout"][name]))
            conditions["parameter_layouts_identical_across_branches_and_before_after"] &= parameter_layout_ok
            conditions["all_parameters_fp32"] &= before.dtype == after.dtype == torch.float32
            conditions["all_saved_parameters_and_gradients_finite"] &= bool(torch.isfinite(before).all() and torch.isfinite(after).all())
            delta = after.reshape(-1).double() - before.reshape(-1).double()
            deltas[branch] = delta
            delta_norm2[branch] += torch.dot(delta, delta).item()
            item = {"before": tensor_identity(before, torch), "after": tensor_identity(after, torch),
                    "initial_equal_to_dt": exact,
                    "before_layout": payload["before_layout"][name],
                    "after_layout": payload["after_layout"][name],
                    "parameter_layout_matches": parameter_layout_ok,
                    "gradient_layout": gradient_layouts[branch][name]}
            if gradient is None:
                none_counts[branch] += 1
                item["gradient"] = None
                native_gradients[branch] = None
            else:
                if not isinstance(gradient, torch.Tensor) or gradient.device.type != "cpu" or gradient.shape != before.shape:
                    raise ValueError("Expected original gradient storage with its native parameter shape.")
                dtype_counts[str(gradient.dtype)] += 1
                conditions["all_non_none_gradients_fp32"] &= gradient.dtype == torch.float32
                conditions["all_saved_parameters_and_gradients_finite"] &= bool(torch.isfinite(gradient).all())
                matched_layout = layout_equal(gradient_layouts[branch][name], payload["before_layout"][name])
                matched_layout &= layout_equal(gradient_layouts[branch][name], gradient_layouts["dt"][name])
                conditions["gradient_layout_matches_parameter_and_all_branches"] &= matched_layout
                item.update(gradient=tensor_identity(gradient, torch), gradient_layout_matches=matched_layout)
                native_gradients[branch] = gradient.reshape(-1).double()
            row["branches"][branch] = item
        present = [native_gradients[branch] is not None for branch in BRANCHES]
        conditions["gradient_none_pattern_identical_across_branches"] &= len(set(present)) == 1
        row["gradient_difference_coordinate_available"] = all(present)
        # None is never silently replaced by a zero gradient. Differences and
        # every reported Gram/dot use only the explicitly common tensor support.
        if all(present):
            comparable_parameters += 1
            comparable_elements += reference.numel()
            for branch in BRANCHES:
                common_delta_norm2[branch] += torch.dot(deltas[branch], deltas[branch]).item()
            vectors = dict(zip(VECTORS[:3], (native_gradients[branch] for branch in BRANCHES)))
            vectors[VECTORS[3]] = native_gradients["dt"] - native_gradients["regularizers_only"]
            vectors[VECTORS[4]] = native_gradients["grpo"] - native_gradients["regularizers_only"]
            for a, b in pairs:
                gram[f"{a}:{b}"] += torch.dot(vectors[a], vectors[b]).item()
            for vector in VECTORS:
                for branch in BRANCHES:
                    dots[vector][branch] += torch.dot(vectors[vector], deltas[branch]).item()
        records.append(row)
    norms = {vector: math.sqrt(gram[f"{vector}:{vector}"]) for vector in VECTORS}
    delta_norms = {branch: math.sqrt(value) for branch, value in delta_norm2.items()}
    common_delta_norms = {branch: math.sqrt(value) for branch, value in common_delta_norm2.items()}
    direction = {vector: {branch: {"dot_gradient_with_actual_delta": dot,
                                  "cosine_gradient_to_actual_delta_on_common_support": cosine(dot, norms[vector], common_delta_norms[branch]),
                                  "dot_negative_gradient_with_actual_delta": -dot}
                          for branch, dot in branch_dots.items()}
                 for vector, branch_dots in dots.items()}
    return {"rank": rank, "scope": "Only this rank's original local shards; no rank aggregation or collective.",
            "arithmetic": "FP64 descriptions of original FP32 storage values; subtraction occurs only in temporary CPU copies.",
            "coordinate_checks": conditions,
            "parameter_count": len(names), "initial_exact_parameter_counts": dict(initial_counts),
            "gradient_none_counts": none_counts, "gradient_dtype_counts": dict(dtype_counts),
            "common_gradient_support": {"parameters": comparable_parameters, "elements": comparable_elements,
                                        "complete_parameter_support": comparable_parameters == len(names)},
            "gradient_Gram_on_common_support": gram, "gradient_norms_on_common_support": norms,
            "gradient_cosines_on_common_support": {f"{a}:{b}": cosine(gram[f"{a}:{b}"], norms[a], norms[b]) for a, b in pairs},
            "actual_parameter_delta_norms": delta_norms,
            "actual_parameter_delta_norms_on_common_gradient_support": common_delta_norms,
            "gradient_projection_on_actual_displacements": direction, "parameters": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run only in the existing CPU environment with CUDA_VISIBLE_DEVICES=''.")
    import torch
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Do not initialize CUDA or a process group for this saved-storage analysis.")
    started, before_resources = time.time(), resources(torch)
    inputs = {}
    completed = read_json(args.input_dir / "completed.json", inputs)
    inspection = read_json(args.input_dir / "native-owner-inspection.json", inputs)
    original_analysis = read_json(args.input_dir / "native-adam-analysis.json", inputs)
    completed_branches = {item["branch"]: item for item in completed["branches"]}
    before_observations = {}
    for branch in BRANCHES:
        path = args.input_dir / f"{branch}-before-logprob.pkl"
        inputs[path.name] = identity(path)
        if inputs[path.name]["sha256"] != completed_branches[branch]["before"]["sha256"]:
            raise ValueError("Actual original before-observation SHA differs from completion receipt.")
        before_observations[branch] = inputs[path.name]
    before_sha_equal = len({item["sha256"] for item in before_observations.values()}) == 1
    rank_results, rank_receipts = [], {}
    for rank in range(2):
        payloads, receipts = {}, {}
        for branch in BRANCHES:
            receipt = read_json(args.input_dir / f"rank{rank}-{branch}-native-adam.json", inputs)
            if receipt["rank"] != rank or receipt["branch"] != branch:
                raise ValueError("Actual native step receipt rank/branch differs from its filename.")
            receipts[branch] = receipt
            path = args.input_dir / f"rank{rank}-{branch}-native-parameter-shards.pt"
            inputs[path.name] = identity(path)
            if inputs[path.name]["sha256"] != receipt["parameter_shards"]["sha256"]:
                raise ValueError("Actual saved native parameter/gradient payload SHA differs from its original receipt.")
            payloads[branch] = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        rank_results.append(rank_geometry(rank, payloads, receipts, torch))
        rank_receipts[str(rank)] = {
            branch: {key: receipt[key] for key in ("sources", "effective_actor_config", "temperature", "multi_turn",
                                                   "optimizer_before", "optimizer_after", "optimizer_step_calls", "scheduler_step_calls")}
            | {"native_preclip_grad_norm": receipt["original_metrics"]["actor/grad_norm"],
               "native_clip_bound": receipt["effective_actor_config"]["grad_clip"],
               "native_preclip_norm_below_bound": all(norm < receipt["effective_actor_config"]["grad_clip"]
                                                     for norm in receipt["original_metrics"]["actor/grad_norm"]),
               "original_reported_kl_loss": receipt["original_metrics"].get("actor/kl_loss")}
            for branch, receipt in receipts.items()}
        rank_results[-1]["recorded_comparison_conditions"] = {
            "effective_actor_config_equal_across_branches": all(receipts[branch]["effective_actor_config"] == receipts["dt"]["effective_actor_config"] for branch in BRANCHES),
            "temperature_and_multi_turn_equal_across_branches": all((receipts[branch]["temperature"], receipts[branch]["multi_turn"]) == (receipts["dt"]["temperature"], receipts["dt"]["multi_turn"]) for branch in BRANCHES),
            "original_reported_kl_scalar_equal_across_branches": all(receipts[branch]["original_metrics"].get("actor/kl_loss") == receipts["dt"]["original_metrics"].get("actor/kl_loss") for branch in BRANCHES),
            "optimizer_before_summary_equal_across_branches": all(receipts[branch]["optimizer_before"] == receipts["dt"]["optimizer_before"] for branch in BRANCHES),
            "optimizer_before_summary_scope": "Native optimizer group/step/key counts; equality of this summary is not independent equality of every historical moment tensor.",
            "all_native_preclip_norms_below_original_bound": all(row["native_preclip_norm_below_bound"] for row in rank_receipts[str(rank)].values()),
        }
        del payloads
    result = {
        "scope": "Saved original three-branch native Adam step only. Per-rank total-gradient differences and actual local parameter displacement; not a reconstructed historical training update.",
        "sources": {"analyzer": identity(__file__), "torch": identity(torch.__file__),
                    "recorded_original_owners": completed["sources"]},
        "inputs": inputs, "source_minibatch_sha256": completed["source_minibatch_sha256"],
        "checkpoint": completed["checkpoint"],
        "configuration_and_comparison_evidence": {
            "original_inspected_actor_config": inspection["actor_config"],
            "before_observation_files_exact_sha_across_branches": before_sha_equal,
            "existing_independent_before_H_LP_array_checks": original_analysis["initial_readout_comparisons"],
            "actual_native_rank_receipts": rank_receipts,
            "rng_and_dropout_scope": "Original complete checkpoint restores are recorded by the original driver. Saved receipts do not independently capture each training-forward RNG state or dropout mask. Identical before LP/H and clipping inactivity alone do not prove that subtracting rounded total gradients equals an exact standalone PG backward.",
        },
        "ranks": rank_results,
        "sign_convention": "For a genuine loss gradient, positive dot(gradient, actual_delta_theta) means a first-order increase in that frozen loss; negative means decrease. Differences are descriptive proxies, not certified exact components. Finite objective changes also include curvature and native storage arithmetic.",
        "interpretation_limits": [
            "DTtotal minus regularizerstotal and GRPOtotal minus regularizerstotal are direct differences of two saved native optimizer-input tensors, after original clipping. Native recorded norms below its bound support inactive clipping; original dtype rounding and separate native backward executions remain part of this measurement.",
            "Common before parameters, inputs, H/LP observations, configurations and unclipped gradients support a matched current-task-gradient interpretation; they do not make cancellation exact. No numeric tolerance or pass/fail threshold is introduced.",
            "regularizers_only restores the same AdamW history and weight decay. Its actual parameter displacement is not a pure current H/KL vector. Parameter-delta differences are never used to linearly attribute Adam contributions.",
            "Each Gram/norm/dot uses only one rank's local shard storage. Old component diagnostic mesh-SUM norms describe a different reduction scope and must not be directly divided by these local norms.",
            "This describes one checkpoint25 saved64 update, not all historical updates, success changes, or whether GRPO is an oracle. Gradient magnitude by itself does not establish task-loss deterioration.",
        ],
        "operations": {"model_initializations": 0, "forwards": 0, "loss_calls": 0,
                       "sampling": 0, "backward": 0, "optimizer_updates": 0, "gathers": 0,
                       "cross_rank_reductions": 0, "Adam_formula_evaluations": 0},
        "runtime": {"started_unix": started, "completed_unix": time.time(),
                    "resources_before": before_resources, "resources_after": resources(torch)},
        "nonfinite_JSON_encoding": "Explicit NaN/+Infinity/-Infinity strings, never numerical pass labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_finite(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = {str(row["rank"]): {"gradient_difference_norms": {key: row["gradient_norms_on_common_support"][key] for key in VECTORS[3:]},
                                 "difference_dot_with_actual_dt_delta": {key: row["gradient_projection_on_actual_displacements"][key]["dt"]["dot_gradient_with_actual_delta"] for key in VECTORS[3:]}}
               for row in rank_results}
    print(json.dumps(json_finite({"output": identity(args.output), "rank_summary": summary}), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
