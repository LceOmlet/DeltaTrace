"""Read the saved original TextCraft minibatch and native gradient receipts on CPU.

Run in the existing VERL/Torch environment, for example::

    CUDA_VISIBLE_DEVICES='' "$VENV_PYTHON" analyze_textcraft_native_minibatch.py OUT

The original DataProto.load_from_disk restores the exact saved object. This
script does not decode tokens, reconstruct trajectories, recalculate credit or
GRPO, run a model, or approximate an Adam update. Statistical reductions use
FP64 copies of saved values; the original arrays and their dtypes are unchanged.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time


LABELS = ("dt_pg", "weighted_entropy", "weighted_kl", "grpo_pg")
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
EXISTING_TOKEN_IDS = (248068, 248069, 271)
TOKEN_MASS_TOP_COUNT = 20


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def source_identity(function):
    result = identity(inspect.getsourcefile(function))
    lines, first = inspect.getsourcelines(function)
    result.update(source_line=first, source_end_line=first + len(lines) - 1)
    return result


def summary(values, torch):
    """Descriptive statistics only; nonfinite values remain explicitly counted."""
    values = values.detach().reshape(-1)
    finite = torch.isfinite(values)
    result = {"count": values.numel(), "finite_count": int(finite.sum()),
              "nan_count": int(torch.isnan(values).sum()),
              "positive_infinity_count": int(torch.isposinf(values).sum()),
              "negative_infinity_count": int(torch.isneginf(values).sum()),
              "input_dtype": str(values.dtype)}
    measured = values[finite].double()
    result["statistics_scope"] = "finite saved values; FP64 descriptive reductions"
    if not measured.numel():
        result.update(nonzero_count=0, positive_count=0, negative_count=0, zero_count=0,
                      mean=None, abs_mean=None, rms=None, signed_sum=None, abs_sum=None,
                      signed_sum_over_abs_sum=None, quantiles=None, abs_quantiles=None)
        return result
    signed_sum, abs_sum = measured.sum().item(), measured.abs().sum().item()
    probabilities = torch.tensor(QUANTILES, dtype=torch.float64)
    result.update(nonzero_count=int(measured.count_nonzero()),
                  positive_count=int((measured > 0).sum()), negative_count=int((measured < 0).sum()),
                  zero_count=int((measured == 0).sum()), mean=measured.mean().item(),
                  abs_mean=measured.abs().mean().item(), rms=measured.square().mean().sqrt().item(),
                  signed_sum=signed_sum, abs_sum=abs_sum,
                  signed_sum_over_abs_sum=signed_sum / abs_sum if abs_sum else None,
                  quantiles={str(q): x for q, x in zip(QUANTILES, torch.quantile(measured, probabilities).tolist())},
                  abs_quantiles={str(q): x for q, x in zip(QUANTILES, torch.quantile(measured.abs(), probabilities).tolist())})
    return result


def token_id_mass_summary(responses, original_mask, dt, grpo, torch):
    """Bounded descriptive mass by saved integer ID; no token interpretation."""
    selected = original_mask != 0
    ids = responses[selected].detach()
    unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    counts = torch.zeros(unique_ids.numel(), dtype=torch.long)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse))
    ids_list = unique_ids.tolist()
    per_method, totals, rankings = {}, {}, {}
    selected_ids = set(EXISTING_TOKEN_IDS)

    for label, advantages in (("dt_actor_advantages", dt), ("official_grpo_advantages", grpo)):
        values = advantages[selected].detach().double()
        finite = torch.isfinite(values)
        positive = finite & (values > 0)
        negative = finite & (values < 0)
        arrays = {}
        for key, source in (
            ("finite_count", finite.long()), ("nonfinite_count", (~finite).long()),
            ("positive_count", positive.long()), ("negative_count", negative.long()),
            ("zero_count", (finite & (values == 0)).long()),
            ("positive_abs_mass", torch.where(positive, values, 0.0)),
            ("negative_abs_mass", torch.where(negative, -values, 0.0)),
        ):
            destination = torch.zeros(unique_ids.numel(), dtype=source.dtype)
            destination.scatter_add_(0, inverse, source)
            arrays[key] = destination
        per_method[label] = arrays
        totals[label] = {key: array.sum().item() for key, array in arrays.items()}
        totals[label].update(signed_mass=totals[label]["positive_abs_mass"] - totals[label]["negative_abs_mass"],
                             abs_mass=totals[label]["positive_abs_mass"] + totals[label]["negative_abs_mass"])
        rankings[label] = {}
        for mass_key in ("positive_abs_mass", "negative_abs_mass"):
            masses = arrays[mass_key]
            # Positive mass is the defining sign, not a numerical threshold.
            order = torch.argsort(masses, descending=True, stable=True)
            order = order[masses[order] > 0][:TOKEN_MASS_TOP_COUNT]
            ranked_ids = unique_ids[order].tolist()
            selected_ids.update(ranked_ids)
            captured, total = masses[order].sum().item(), totals[label][mass_key]
            rankings[label][mass_key] = {"token_ids": ranked_ids, "captured_mass": captured,
                                        "total_mass": total, "coverage": captured / total if total else None}

    index_by_id = {token_id: index for index, token_id in enumerate(ids_list)}
    selected_indices = [index_by_id[token_id] for token_id in sorted(selected_ids) if token_id in index_by_id]
    records = []
    for token_id in sorted(selected_ids):
        index = index_by_id.get(token_id)
        count = counts[index].item() if index is not None else 0
        record = {"token_id": token_id, "action_token_count": count,
                  "requested_existing_id": token_id in EXISTING_TOKEN_IDS}
        for label, arrays in per_method.items():
            stats = {key: array[index].item() if index is not None else 0 for key, array in arrays.items()}
            stats.update(signed_mass=stats["positive_abs_mass"] - stats["negative_abs_mass"],
                         abs_mass=stats["positive_abs_mass"] + stats["negative_abs_mass"])
            stats["mean_abs_finite"] = stats["abs_mass"] / stats["finite_count"] if stats["finite_count"] else None
            stats["signed_mean_finite"] = stats["signed_mass"] / stats["finite_count"] if stats["finite_count"] else None
            stats["zero_fraction_finite"] = stats["zero_count"] / stats["finite_count"] if stats["finite_count"] else None
            record[label] = stats
        records.append(record)
    coverage = {}
    for label, arrays in per_method.items():
        coverage[label] = {}
        for key in ("positive_abs_mass", "negative_abs_mass"):
            captured, total = arrays[key][selected_indices].sum().item(), totals[label][key]
            coverage[label][key] = {"captured_mass": captured, "total_mass": total,
                                    "coverage": captured / total if total else None}
    return {
        "scope": "Global pooled saved actor-action token occurrences, grouped by original integer responses ID; no decoding, re-encoding or token-function inference.",
        "weighting_limit": "Pooled mass/frequency is descriptive and is not the native PPO loss weighting: the original actor averages each B4 over its own loss_mask token count, then accumulates eight B4 losses with /8.",
        "selection": "Union of each method's top20 positive/negative finite absolute mass IDs, plus the three explicitly requested existing IDs.",
        "requested_existing_ids": list(EXISTING_TOKEN_IDS), "requested_id_meaning": "ID categories only; this statistic does not assign a function to these IDs.",
        "action_token_count": ids.numel(), "unique_action_token_ids": len(ids_list),
        "selected_id_action_token_count": int(counts[selected_indices].sum()),
        "finite_mass_totals": totals, "top20_rankings": rankings, "union_mass_coverage": coverage,
        "token_id_records": records,
        "nonfinite_scope": "Nonfinite advantages are counted by ID and excluded from mass statistics; no replacement advantage is supplied to a loss.",
        "zero_total_mass_coverage": "null when the corresponding observed total mass is zero; no epsilon or pass threshold is introduced.",
    }


def batch_summary(data, torch):
    tensors = data.batch
    responses = tensors["responses"]
    rows, width = responses.shape
    # These are the original actor's mask branches, not a reconstructed mask.
    multi_turn = bool(data.meta_info.get("multi_turn", False))
    mask_name = "loss_mask" if multi_turn else "attention_mask"
    original_mask = tensors[mask_name][:, -width:]
    selected = original_mask != 0
    dt = tensors["advantages"]
    grpo = tensors["diagnostic_grpo_advantages"]
    if dt.shape != responses.shape or grpo.shape != responses.shape or selected.shape != responses.shape:
        raise ValueError("Saved advantages and the original actor mask must bind the saved response columns.")
    # This is the official GRPO owner's saved-score reduction, not an episode
    # parser or a reconstruction of the per-turn complete-return calculation.
    rewards = tensors["token_level_rewards"]
    returns = rewards.sum(dim=-1)
    uid = data.non_tensor_batch["uid"].tolist()
    trajectory_uid = data.non_tensor_batch["traj_uid"].tolist()
    groups = OrderedDict()
    for row, group_uid in enumerate(uid):
        groups.setdefault(str(group_uid), []).append(row)

    def signals(row_indices):
        indices = torch.tensor(row_indices, dtype=torch.long)
        action_mask = selected[indices]
        result = {"row_count": len(row_indices), "action_token_count": int(action_mask.sum()),
                  "dt_actor_advantages": summary(dt[indices][action_mask], torch),
                  "official_grpo_advantages": summary(grpo[indices][action_mask], torch)}
        if "dt_q_estimates" in tensors:
            q_values = tensors["dt_q_estimates"][indices][action_mask]
            result["saved_dt_q_counts"] = {"positive": int((q_values > 0).sum()),
                                           "zero": int((q_values == 0).sum()),
                                           "negative": int((q_values < 0).sum()),
                                           "nonfinite": int((~torch.isfinite(q_values)).sum())}
        return result

    finite_returns = torch.isfinite(returns)
    return_masks = {"return_positive": finite_returns & (returns > 0),
                    "return_zero": finite_returns & (returns == 0),
                    "return_negative": finite_returns & (returns < 0), "return_nonfinite": ~finite_returns}
    by_return = {}
    for label, row_mask in return_masks.items():
        indices = row_mask.nonzero(as_tuple=True)[0].tolist()
        by_return[label] = dict(signals(indices), row_indices=indices,
                                saved_training_returns=summary(returns[row_mask], torch))
    group_records = []
    for group_uid, indices in groups.items():
        group_returns = returns[indices]
        record = dict(uid=group_uid, row_indices=indices,
                      trajectory_uid=[str(trajectory_uid[i]) for i in indices],
                      saved_training_returns=group_returns.tolist(),
                      return_positive_rows=int((group_returns > 0).sum()),
                      return_zero_rows=int((group_returns == 0).sum()),
                      return_negative_rows=int((group_returns < 0).sum()), **signals(indices))
        record["by_saved_training_return"] = {
            label: {"row_count": int(row_mask[indices].sum()),
                    "action_token_count": int(selected[indices][row_mask[indices]].sum())}
            for label, row_mask in return_masks.items()}
        group_records.append(record)
    result = {
        "scope": "All saved global minibatch rows and original actor policy-token mask; no token decoding or trajectory reconstruction.",
        "rows": rows, "uid_groups": len(groups), "unique_trajectory_uid": len(set(map(str, trajectory_uid))),
        "uid_group_row_counts": [len(indices) for indices in groups.values()],
        "meta_info": {key: data.meta_info.get(key) for key in ("multi_turn", "temperature")},
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                   for key, value in tensors.items()},
        "mask": {"owner_field": mask_name, "selection": f"{mask_name}[:, -responses.shape[1]:] != 0",
                 "action_token_count": int(selected.sum()), "original_mask_sum": original_mask.sum().item(),
                 "original_mask_values": torch.unique(original_mask).tolist(),
                 "response_mask_token_count": int((tensors["response_mask"] != 0).sum())
                 if "response_mask" in tensors else None},
        "return_definition": "Original token_level_rewards.sum(dim=-1), as used by the official GRPO owner; this row score is distinct from each DT source response's complete future G_t.",
        "all_action_tokens": signals(list(range(rows))), "by_saved_training_return": by_return,
        "uid_group_statistics": group_records,
        "outside_actor_mask": {"dt_actor_advantages": summary(dt[~selected], torch),
                               "official_grpo_advantages": summary(grpo[~selected], torch)},
        "grpo_definition": "Read saved diagnostic_grpo_advantages, computed before original dispatch by the official global-UID GRPO owner; not recomputed by this analyzer.",
        "original_token_id_mass": token_id_mass_summary(responses, original_mask, dt, grpo, torch),
    }
    if "dt_token_advantages" in tensors:
        source = tensors["dt_token_advantages"]
        result["dt_actor_binding"] = {"torch_equal_saved_dt_token_advantages": torch.equal(dt, source),
                                      "dt_token_advantages_dtype": str(source.dtype)}
    else:
        result["dt_actor_binding"] = {"missing_field": "dt_token_advantages"}
    if "dt_q_estimates" in tensors:
        q = tensors["dt_q_estimates"]
        # Q_hat is the actual observed G_t in the accepted owner construction.
        # Count its saved values only; do not infer them from advantage signs.
        result["saved_dt_q_action_tokens"] = {
            "definition": "Saved dt_q_estimates on original actor policy-token positions; zero also includes any owner-unmapped positions, not a reconstructed event label.",
            "statistics": summary(q[selected], torch), "positive_count": int(((q > 0) & selected).sum()),
            "zero_count": int(((q == 0) & selected).sum()), "negative_count": int(((q < 0) & selected).sum()),
            "nonfinite_count": int((~torch.isfinite(q) & selected).sum()),
        }
    else:
        result["saved_dt_q_action_tokens"] = {"missing_field": "dt_q_estimates"}
    return result


def gradient_summary(record):
    """Compose recorded gradient inner products, never optimizer dynamics."""
    statistics = record["gradient_statistics"]
    inner_products = statistics["inner_products"]
    vectors = {label: {label: 1.0} for label in LABELS}
    vectors.update(regularizer={"weighted_entropy": 1.0, "weighted_kl": 1.0},
                   dt_total_loss={"dt_pg": 1.0, "weighted_entropy": 1.0, "weighted_kl": 1.0},
                   grpo_total_loss={"grpo_pg": 1.0, "weighted_entropy": 1.0, "weighted_kl": 1.0})
    undefined = []

    def dot(a, b):
        total = 0.0
        for label_a, coefficient_a in vectors[a].items():
            for label_b, coefficient_b in vectors[b].items():
                key = f"{label_a}:{label_b}"
                if key not in inner_products:
                    key = f"{label_b}:{label_a}"
                total += coefficient_a * coefficient_b * inner_products[key]
        return total

    squared_norms = {label: dot(label, label) for label in vectors}
    norms = {}
    for label, squared in squared_norms.items():
        if math.isfinite(squared) and squared >= 0:
            norms[label] = math.sqrt(squared)
        else:
            norms[label] = None
            undefined.append({"metric": f"norm:{label}", "reason": "nonfinite or negative recorded Gram quadratic form", "squared_norm": squared})

    def ratio(numerator, denominator):
        a, b = norms[numerator], norms[denominator]
        key = f"{numerator}/{denominator}"
        if a is None or b is None or b == 0:
            undefined.append({"metric": key, "reason": "nonfinite norm or zero denominator norm"})
            return None
        return a / b

    pairs = [(a, b) for i, a in enumerate(LABELS) for b in LABELS[i + 1:]]
    pairs += [("regularizer", "dt_pg"), ("regularizer", "grpo_pg"),
              ("dt_total_loss", "grpo_pg"), ("dt_total_loss", "grpo_total_loss")]
    relationships = {}
    for a, b in pairs:
        product, norm_a, norm_b = dot(a, b), norms[a], norms[b]
        cosine, angle = None, None
        if math.isfinite(product) and norm_a is not None and norm_b is not None and norm_a and norm_b:
            cosine = product / (norm_a * norm_b)
            # Leave roundoff outside [-1,1] visible; no invented clipping/tolerance.
            if -1 <= cosine <= 1:
                angle = math.degrees(math.acos(cosine))
            else:
                undefined.append({"metric": f"angle:{a}:{b}", "reason": "recorded scalar cosine outside acos domain", "cosine": cosine})
        else:
            undefined.append({"metric": f"cosine:{a}:{b}", "reason": "nonfinite inner product/norm or zero norm"})
        relationships[f"{a}:{b}"] = {"inner_product": product, "cosine": cosine, "angle_degrees": angle}
    ratio_pairs = [("weighted_entropy", "dt_pg"), ("weighted_kl", "dt_pg"),
                   ("regularizer", "dt_pg"), ("dt_pg", "grpo_pg"),
                   ("weighted_entropy", "grpo_pg"), ("weighted_kl", "grpo_pg"),
                   ("regularizer", "grpo_pg"), ("dt_total_loss", "grpo_pg"),
                   ("dt_total_loss", "grpo_total_loss")]
    losses = {}
    for label in LABELS:
        observed = record["passes"][label]
        component = "dt_pg" if label == "grpo_pg" else label
        selected_scalars = [micro[component] for micro in observed["microbatch_losses"]]
        native_scale = record["effective_config"]["native_backward_scale"]
        losses[label] = {
            "microbatch_count": len(selected_scalars),
            "selected_native_scalar_values": [item["value"] for item in selected_scalars],
            "native_coefficients": [item["native_coefficient"] for item in selected_scalars],
            "native_accumulated_selected_loss": sum(item["value"] * item["native_coefficient"] * native_scale for item in selected_scalars),
            "optimizer_boundary": observed["optimizer_boundary"],
            "original_metrics": observed["original_metrics"],
        }
    return {
        "rank": record["rank"], "scope": record["scope"], "sources": record["sources"],
        "effective_config": record["effective_config"],
        "optimizer_step_executed": record["optimizer_step_executed"],
        "native_gradient_clipping_executed": record["native_gradient_clipping_executed"],
        "gradient_statistics_scope": statistics["norm_scope"],
        "measurement_dtype": statistics["measurement_dtype"], "ownership_buckets": statistics["ownership_buckets"],
        "component_weighting": "Recorded gradients already contain original entropy/KL coefficients and original accumulation scale; neither is applied again.",
        "composite_definitions": {key: value for key, value in vectors.items() if key not in LABELS},
        "composite_measurement_scope": "Algebraically derived from the four recorded separate gradient passes and their Gram inner products; no fifth combined backward was measured.",
        "norms": norms, "squared_norms": squared_norms, "original_recorded_norms": statistics["norms"],
        "original_inner_products": inner_products,
        "norm_ratios": {f"{a}/{b}": ratio(a, b) for a, b in ratio_pairs},
        "relationships": relationships, "selected_native_losses": losses, "undefined_metrics": undefined,
        "direction_scope": "Loss gradients before native clipping. Simultaneous sign reversal to negative-gradient descent directions preserves these cosines; this does not measure an Adam update or prove causation from a negative cosine.",
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Existing native minibatch diagnostic OUT directory")
    parser.add_argument("--output", type=Path, help="Default: OUT/native-minibatch-analysis.json")
    args = parser.parse_args()
    # Establish CPU-only restore before the official protocol imports Torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from verl.protocol import DataProto

    directory = args.directory.resolve()
    output = (args.output or directory / "native-minibatch-analysis.json").resolve()
    paths = {"minibatch": directory / "native-optimizer-minibatch.pkl",
             **{f"rank{rank}_gradients": directory / f"rank{rank}-minibatch-gradients.json" for rank in (0, 1)}}
    inputs = {name: identity(path) for name, path in paths.items()}
    data = DataProto.load_from_disk(str(paths["minibatch"]))
    if any(tensor.device.type != "cpu" for tensor in data.batch.values()):
        raise RuntimeError("Original DataProto restore was not CPU-only; no GPU tensor transfer or substitute loader is performed.")
    records = {str(rank): json.loads(paths[f"rank{rank}_gradients"].read_bytes()) for rank in (0, 1)}
    sources = {"analyzer": identity(__file__), "data_proto_load_from_disk": source_identity(DataProto.load_from_disk),
               "data_proto_setstate": source_identity(DataProto.__setstate__), "torch": identity(torch.__file__)}
    context = {}
    runtime_context = {}
    for name in ("prepared-diagnostic.json", "job.json", "launch.json", "native-minibatch-source-identity.json",
                 "native-minibatch-effective-config.json", "native-minibatch-update-boundary.json", "native-minibatch-completed.json"):
        path = directory / name
        if path.is_file():
            context[name] = identity(path)
            if name in ("prepared-diagnostic.json", "job.json", "native-minibatch-update-boundary.json"):
                saved = json.loads(path.read_bytes())
                runtime_context[name] = {key: saved[key] for key in
                    ("role", "source_kind", "checkpoint", "devices", "pid", "pid_birth", "started_unix",
                     "global_step", "rows", "groups", "snapshot", "grpo_owner") if key in saved}
                if name == "native-minibatch-update-boundary.json" and "snapshot" in saved:
                    runtime_context[name]["recorded_snapshot_sha_matches_input"] = (
                        saved["snapshot"]["sha256"] == inputs["minibatch"]["sha256"])
    result = {
        "scope": "Read-only CPU statistics of one real native complete minibatch and its recorded four loss gradients; not a training replay, model evaluation, quality gate, or optimizer update.",
        "created_unix": time.time(), "pid": os.getpid(), "torch_version": torch.__version__,
        "inputs": inputs, "sources": sources, "context_receipts": context,
        "saved_runtime_context": runtime_context,
        "native_minibatch": batch_summary(data, torch),
        "gradient_ranks": {rank: gradient_summary(record) for rank, record in records.items()},
        "rank_comparison": {
            "scope": "Compare already-recorded per-rank scalar statistics; no sum across ranks, because sharded statistics already use their original mesh SUM.",
            "effective_config_equal": records["0"]["effective_config"] == records["1"]["effective_config"],
            "inner_product_rank1_minus_rank0": {
                key: records["1"]["gradient_statistics"]["inner_products"][key] - value
                for key, value in records["0"]["gradient_statistics"]["inner_products"].items()},
        },
        "limits": ["One newly collected checkpoint-local minibatch does not reconstruct the historical first bad update.",
                   "A row with saved training return zero is distinct from each DT source response's complete future G_t; both saved row scores and saved Q token counts are reported.",
                   "Signed cancellation and gradient angles are observations, not a correctness proof or causal diagnosis.",
                   "No Adam state, parameter update, new loss, custom numerical threshold, or token/environment parser is constructed."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "analyzed_saved_native_artifacts", "output": identity(output),
                      "rows": result["native_minibatch"]["rows"], "uid_groups": result["native_minibatch"]["uid_groups"]}))


if __name__ == "__main__":
    main()
