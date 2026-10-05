"""CPU-only descriptive reductions of the original saved native64 DataProto.

No model, forward, credit generation, backward, optimizer, or GPU call.  The
original protocol restores the carrier and splits native DP chunks.  The pinned
core owner supplies all token-mean aggregation and same-policy PG scalar calls.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import time


ACTOR_SHA = "1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd"
CORE_SHA = "fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299"
PROTO_SHA = "2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa"
SNAPSHOT_SHA = "45ae51e3e18f206e238a1f1e934eaf9a4074c9361dabdab83bb3e00a2b4b084d"


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def function_identity(function):
    result = identity(inspect.getsourcefile(function))
    lines, start = inspect.getsourcelines(function)
    result.update(source_line=start, source_end_line=start + len(lines) - 1)
    return result


def require_identity(record, expected):
    if record["sha256"] != expected:
        raise ValueError("Recorded native source/input identity changed: " + record["path"])


def resource_snapshot(psutil):
    process = psutil.Process()
    memory = process.memory_full_info()
    return {"rss_bytes": memory.rss, "pss_bytes": getattr(memory, "pss", None),
            "host_available_bytes": psutil.virtual_memory().available}


def descriptive(values):
    values = values.detach().double().reshape(-1)
    finite = values.isfinite()
    if not bool(finite.all()):
        raise ValueError("This saved carrier no longer has the recorded finite advantages")
    count = values.numel()
    if not count:
        return {"count": 0, "mean": None, "abs_mean": None, "rms": None,
                "positive_count": 0, "negative_count": 0, "zero_count": 0,
                "signed_mass": 0.0, "positive_mass": 0.0, "negative_abs_mass": 0.0,
                "abs_mass": 0.0, "signed_over_abs_mass": None}
    positive = values[values > 0].sum().item()
    negative = -values[values < 0].sum().item()
    absolute = positive + negative
    return {"count": count, "mean": values.mean().item(),
            "abs_mean": values.abs().mean().item(),
            "rms": values.square().mean().sqrt().item(),
            "positive_count": int((values > 0).sum()),
            "negative_count": int((values < 0).sum()), "zero_count": int((values == 0).sum()),
            "signed_mass": values.sum().item(), "positive_mass": positive,
            "negative_abs_mass": negative, "abs_mass": absolute,
            "signed_over_abs_mass": (positive - negative) / absolute if absolute else None}


def original_denominator_contributions(values, original_mask, subset, torch, aggregate):
    """Zero other numerators; retain the full original B4 loss_mask denominator."""
    selected = torch.where(subset, values, torch.zeros_like(values))
    call = lambda matrix: aggregate(loss_mat=matrix, loss_mask=original_mask,
                                    loss_agg_mode="token-mean").item()
    return {"mean": call(selected), "abs_mean": call(selected.abs()),
            "mean_square": call(selected.square()),
            "positive_mass_per_original_denominator": call(selected.clamp_min(0)),
            "negative_abs_mass_per_original_denominator": call((-selected).clamp_min(0))}


def same_policy_scalar(micro, mask, advantages, actor_config, core):
    """Original owner loss at ratio=1; no native forward or logprob reconstruction."""
    result = core.compute_policy_loss(
        old_log_prob=micro["old_log_probs"], log_prob=micro["old_log_probs"],
        advantages=advantages, response_mask=mask,
        cliprange=actor_config["clip_ratio"],
        cliprange_low=actor_config["clip_ratio_low"],
        cliprange_high=actor_config["clip_ratio_high"],
        clip_ratio_c=actor_config["clip_ratio_c"],
        loss_agg_mode=actor_config["loss_agg_mode"])
    return {key: value.item() for key, value in zip(
        ("pg_loss", "pg_clipfrac", "ppo_kl", "pg_clipfrac_lower"), result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_out", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run only with the existing environment and CUDA_VISIBLE_DEVICES=''")
    import psutil
    import torch
    from verl import DataProto
    from verl.trainer.ppo import core_algos as core

    before = resource_snapshot(psutil)
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise ValueError("This diagnostic must not inherit an initialized GPU/process group")
    native = args.native_out
    boundary_path = native / "native-minibatch-update-boundary.json"
    boundary = json.loads(boundary_path.read_bytes())
    config_path = native / "native-minibatch-effective-config.json"
    config = json.loads(config_path.read_bytes())
    actor_config = config["actor_rollout_ref"]["actor"]
    if actor_config["shuffle"] or actor_config["use_dynamic_bsz"] or actor_config["ppo_epochs"] != 1:
        raise ValueError("Not the recorded continuous static B4 actor path")
    if actor_config["ppo_mini_batch_size"] != 64 or actor_config["ppo_micro_batch_size_per_gpu"] != 4:
        raise ValueError("Not the recorded global64/local32/B4 configuration")
    if actor_config["loss_agg_mode"] != "token-mean" or actor_config["policy_loss"]["loss_mode"] != "vanilla":
        raise ValueError("Not the recorded official vanilla token-mean loss")
    input_record = identity(boundary["snapshot"]["path"])
    require_identity(input_record, boundary["snapshot"]["sha256"])
    require_identity(input_record, SNAPSHOT_SHA)
    sources = {"analyzer": identity(__file__), "protocol_load": function_identity(DataProto.load_from_disk),
               "protocol_chunk": function_identity(DataProto.chunk),
               "policy_loss": function_identity(core.compute_policy_loss),
               "loss_aggregation": function_identity(core.agg_loss),
               "torch": identity(torch.__file__)}
    require_identity(sources["protocol_load"], PROTO_SHA)
    require_identity(sources["protocol_chunk"], PROTO_SHA)
    require_identity(sources["policy_loss"], CORE_SHA)
    require_identity(sources["loss_aggregation"], CORE_SHA)
    saved_gradient_paths = [native / f"rank{rank}-minibatch-gradients.json" for rank in range(2)]
    saved_gradients = [json.loads(path.read_bytes()) for path in saved_gradient_paths]
    sources["actor"] = identity(saved_gradients[0]["sources"]["actor"]["path"])
    require_identity(sources["actor"], ACTOR_SHA)
    for saved in saved_gradients:
        effective = saved["effective_config"]
        if effective["ppo_mini_batch_size"] != 32 or effective["gradient_accumulation"] != 8:
            raise ValueError("Saved backward owner no longer records native local32/B4/8 accumulation")
    data = DataProto.load_from_disk(str(input_record["path"]))
    if len(data) != 64 or not data.meta_info.get("multi_turn"):
        raise ValueError("Not the exact original multi-turn native64 carrier")
    if any(value.device.type != "cpu" for value in data.batch.values()):
        raise ValueError("Original saved batch must be entirely CPU tensors")
    uid = data.non_tensor_batch["uid"].tolist()
    trajectory_uid = data.non_tensor_batch["traj_uid"].tolist()
    if uid != boundary["uid"] or trajectory_uid != boundary["traj_uid"]:
        raise ValueError("Original update-boundary row identities changed")
    if not torch.equal(data.batch["advantages"], data.batch["dt_token_advantages"]):
        raise ValueError("Actual actor and DT advantage fields differ")
    keys = ("responses", "loss_mask", "old_log_probs", "advantages",
            "diagnostic_grpo_advantages", "dt_q_estimates", "token_level_rewards")
    local_chunks = data.chunk(chunks=2)
    records, rank_summaries, row_records = [], [], []
    with torch.no_grad():
        for rank, local in enumerate(local_chunks):
            if len(local) != 32 or local.non_tensor_batch["traj_uid"].tolist() != trajectory_uid[rank * 32:(rank + 1) * 32]:
                raise ValueError("Original protocol did not produce the recorded continuous DP chunks")
            tensor_batch = local.batch.select(*keys)
            mini_batches = tensor_batch.split(32)
            if len(mini_batches) != 1:
                raise ValueError("Not one native optimizer minibatch per rank")
            micros = mini_batches[0].split(4)
            for index, micro in enumerate(micros):
                width = micro["responses"].shape[1]
                mask = micro["loss_mask"][:, -width:]
                selected = mask != 0
                if torch.unique(mask).tolist() not in ([0, 1], [1]):
                    raise ValueError("Original action mask is no longer the recorded binary mask")
                denominator = mask.sum().item()
                if not denominator:
                    raise ValueError("An original B4 has no actor tokens")
                returns = micro["token_level_rewards"].sum(dim=-1)
                if not bool(((returns == 0) | (returns == 1)).all()):
                    raise ValueError("Saved TextCraft row returns are no longer the recorded 0/1 values")
                q = micro["dt_q_estimates"]
                subsets = {"all": torch.ones_like(selected), "return_positive": (returns > 0)[:, None].expand_as(selected),
                           "return_zero": (returns == 0)[:, None].expand_as(selected),
                           "Q_positive": q > 0, "Q_zero": q == 0}
                methods = {"DT": micro["advantages"], "GRPO": micro["diagnostic_grpo_advantages"]}
                record = {"rank": rank, "microbatch_index": index,
                          "global_row_indices": list(range(rank * 32 + index * 4, rank * 32 + index * 4 + 4)),
                          "original_mask_denominator": denominator, "native_accumulation_divisor": 8,
                          "row_returns": returns.tolist(), "support": {}, "methods": {}, "saved_native_losses": {}}
                for label, subset in subsets.items():
                    count = int((selected & subset).sum())
                    record["support"][label] = {"action_tokens": count, "fraction_of_original_denominator": count / denominator}
                for label, values in methods.items():
                    record["methods"][label] = {name: {
                        "conditional_support_descriptive_FP64": descriptive(values[selected & subset]),
                        "original_B4_denominator_native_FP32": original_denominator_contributions(values, mask, subset, torch, core.agg_loss)}
                        for name, subset in subsets.items()}
                    replay = same_policy_scalar(micro, mask, values, actor_config, core)
                    pass_name = "dt_pg" if label == "DT" else "grpo_pg"
                    saved_micro = saved_gradients[rank]["passes"][pass_name]["microbatch_losses"][index]
                    saved_loss = saved_micro["dt_pg"]["value"]
                    if saved_micro["microbatch_index"] != index:
                        raise ValueError("Saved native B4 scalar order changed")
                    record["methods"][label]["same_policy_scalar_replay"] = {
                        **replay, "saved_native_pg_loss": saved_loss,
                        "difference_from_saved_native_pg_loss": replay["pg_loss"] - saved_loss,
                        "exact_scalar_equal": replay["pg_loss"] == saved_loss,
                        "saved_native_ppo_kl": saved_micro["ppo_kl"],
                        "saved_native_clipfrac": saved_micro["pg_clipfrac"],
                        "saved_native_clipfrac_lower": saved_micro["pg_clipfrac_lower"]}
                for pass_name in ("dt_pg", "weighted_entropy", "weighted_kl", "grpo_pg"):
                    saved_micro = saved_gradients[rank]["passes"][pass_name]["microbatch_losses"][index]
                    record["saved_native_losses"][pass_name] = {
                        field: saved_micro[field] for field in ("dt_pg", "weighted_entropy", "weighted_kl")}
                for row in range(4):
                    global_row = rank * 32 + index * 4 + row
                    row_records.append({"global_row": global_row, "uid": uid[global_row],
                        "traj_uid": trajectory_uid[global_row], "rank": rank, "microbatch_index": index,
                        "saved_row_return": returns[row].item(), "action_tokens": int(selected[row].sum()),
                        "Q_zero_action_tokens": int((selected[row] & (q[row] == 0)).sum()),
                        "DT": descriptive(methods["DT"][row][selected[row]]),
                        "GRPO": descriptive(methods["GRPO"][row][selected[row]])})
                records.append(record)
            own = [record for record in records if record["rank"] == rank]
            rank_summary = {"rank": rank, "microbatches": len(own), "accumulation_divisor": 8,
                "original_action_tokens": sum(record["original_mask_denominator"] for record in own),
                "B4_token_mean_then_divide8_support_fractions": {
                    name: sum(record["support"][name]["fraction_of_original_denominator"] for record in own) / 8
                    for name in subsets}, "methods": {}}
            for label in methods:
                rank_summary["methods"][label] = {name: {
                    key: sum(record["methods"][label][name]["original_B4_denominator_native_FP32"][key] for record in own) / 8
                    for key in own[0]["methods"][label][name]["original_B4_denominator_native_FP32"]}
                    for name in subsets}
                rank_summary["methods"][label]["same_policy_pg_loss_over8"] = sum(
                    record["methods"][label]["same_policy_scalar_replay"]["pg_loss"] for record in own) / 8
                rank_summary["methods"][label]["saved_native_pg_loss_over8"] = sum(
                    record["methods"][label]["same_policy_scalar_replay"]["saved_native_pg_loss"] for record in own) / 8
            rank_summaries.append(rank_summary)
    across = {"scope": "Arithmetic mean of two recorded rank scalar summaries;16 equally weighted B4, not global pooled token mean or gradient norm.",
        "support_fractions": {name: sum(rank["B4_token_mean_then_divide8_support_fractions"][name] for rank in rank_summaries) / 2 for name in subsets},
        "methods": {label: {name: {key: sum(rank["methods"][label][name][key] for rank in rank_summaries) / 2
                    for key in rank_summaries[0]["methods"][label][name]} for name in subsets} for label in methods}}
    for label in methods:
        across["methods"][label]["same_policy_pg_loss"] = sum(rank["methods"][label]["same_policy_pg_loss_over8"] for rank in rank_summaries) / 2
        across["methods"][label]["saved_native_pg_loss"] = sum(rank["methods"][label]["saved_native_pg_loss_over8"] for rank in rank_summaries) / 2
    total_tokens = sum(record["original_mask_denominator"] for record in records)
    total_Q_zero = sum(record["support"]["Q_zero"]["action_tokens"] for record in records)
    comparison = {"scalar_comparisons": 32,
        "exact_equal_count": sum(record["methods"][label]["same_policy_scalar_replay"]["exact_scalar_equal"] for record in records for label in methods),
        "max_abs_difference": max(abs(record["methods"][label]["same_policy_scalar_replay"]["difference_from_saved_native_pg_loss"]) for record in records for label in methods),
        "scope": "CPU original policy-loss scalar at stored old==old versus previously measured native scalar. Descriptive exact/difference comparison; no tolerance or numerical pass gate."}
    after = resource_snapshot(psutil)
    result = {"scope": "Saved native64 CPU carrier reductions; original protocol DP chunk2/local32/static B4 and original owner token-mean/8. No model or state update.",
        "created_unix": time.time(), "sources": sources,
        "inputs": {"snapshot": input_record, "update_boundary": identity(boundary_path),
                   "effective_config": identity(config_path), "rank_gradients": [identity(path) for path in saved_gradient_paths]},
        "contracts": {"owner_actor_lines": {"continuous_microbatch_split": [399, 405], "original_mask": [412, 417], "PG_H_KL": [443, 469], "accumulation": [473, 478]},
            "global_rows": 64, "DP_ranks": 2, "local_rows": 32, "micro_rows": 4, "accumulation": 8,
            "actual_config": actor_config, "actual_native_local_config": saved_gradients[0]["effective_config"],
            "return_scope": "Original token_level_rewards.sum(-1), same row score as official GRPO; Q_zero is separately selected from actual saved dt_q_estimates.",
            "conditional_scope": "Conditional subset statistics use that subset's count only for descriptive successful-token amplitude; they never replace the original loss denominator.",
            "same_policy_scope": "Original core policy loss called with saved old_log_probs for both old and current, deliberately setting ratio=1. No current logits reconstructed. Saved native PPO KL/clipfrac are reported separately; mean PPO KL=0 alone does not prove every historical/native token ratio is1.",
            "statistics_dtype": "Original owner scalar reductions remain FP32; conditional descriptive masses/moments use temporary FP64 reductions only."},
        "coverage": {"rows": len(row_records), "microbatches": len(records), "original_action_tokens": total_tokens,
            "Q_zero_action_tokens": total_Q_zero, "global_pooled_Q_zero_fraction": total_Q_zero / total_tokens,
            "native_B4_Q_zero_fraction_mean": across["support_fractions"]["Q_zero"],
            "success_rows": sum(row["saved_row_return"] > 0 for row in row_records),
            "zero_return_rows": sum(row["saved_row_return"] == 0 for row in row_records),
            "B4_with_zero_Q_positive_tokens": sum(record["support"]["Q_positive"]["action_tokens"] == 0 for record in records)},
        "same_policy_saved_scalar_comparison": comparison, "B4": records,
        "rank_summaries": rank_summaries, "across_ranks": across, "rows": row_records,
        "runtime": {"pid": os.getpid(), "pid_birth": psutil.Process().create_time(),
            "python": os.sys.executable, "torch_version": torch.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_initialized": torch.cuda.is_initialized(), "distributed_initialized": torch.distributed.is_initialized(),
            "started_unix": started, "elapsed_seconds": time.time() - started,
            "before": before, "after": after, "maxRSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "operations": {"model_loads": 0, "model_forwards": 0, "DT_calls": 0, "backward_calls": 0, "optimizer_steps": 0, "scheduler_steps": 0},
        "limits": ["Mass, sign cancellation and scalar-loss values are not gradient norms or gradient directions.",
            "Subset original-denominator statistics preserve the existing B4 denominator; none is a suggested credit normalization.",
            "GRPO is a same-data official advantage comparison, not an oracle or a guaranteed stable trajectory.",
            "One checkpoint25 diagnostic minibatch does not reconstruct all historical actor updates.",
            "No entropy tensor/logit forward is available or generated here; original recorded H/KL scalar observations are only copied for comparison."]}
    require_identity(identity(input_record["path"]), input_record["sha256"])
    if result["runtime"]["cuda_initialized"] or result["runtime"]["distributed_initialized"]:
        raise ValueError("CPU-only lifecycle contract failed")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": identity(args.output), "coverage": result["coverage"],
        "comparison": comparison, "across_ranks": across, "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
