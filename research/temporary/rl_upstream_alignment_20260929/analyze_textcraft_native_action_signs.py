"""Observe saved original native Adam LP changes by saved return/DT-A support.

CPU only: original DataProto load/chunk and original core own the loss/reduction.
The original full B4 mask is retained for every loss partition. No model, credit
recomputation, forward, backward, optimizer, rollout or trajectory parser runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from analyze_textcraft_native_adam import identity, source, resources, BRANCHES, PROTO_SHA, CORE_SHA


MINIBATCH_SHA = "45ae51e3e18f206e238a1f1e934eaf9a4074c9361dabdab83bb3e00a2b4b084d"
HELPER_SHA = "318a994fe35f68ae6400c76c1e735973ed1955771918677213f0bed92be792d1"
COMPLETED_SHA = "363a868f2d7aa4162af27d1f7e026b29ba91cb0d0a8c5c5796924717b859f653"
INSPECTION_SHA = "c965d0b88d070a324f9823712e22a9b58c41551273397592b87c1071913625b3"
OBJECTIVE_SHA = "1da0fabf7596f2446703ac582e4c16c76b4571045852c3553c0681a3e1244733"


def descriptive(values, selected, torch):
    value = values[selected].detach().double()
    finite = torch.isfinite(value)
    measured = value[finite]
    result = {"count": value.numel(), "finite_count": measured.numel(),
              "nonfinite_count": int((~finite).sum())}
    if not measured.numel():
        result.update(mean=None, abs_mean=None, RMS=None, minimum=None, maximum=None,
                      positive_count=0, negative_count=0, zero_count=0)
    else:
        result.update(mean=measured.mean().item(), abs_mean=measured.abs().mean().item(),
                      RMS=measured.square().mean().sqrt().item(), minimum=measured.min().item(), maximum=measured.max().item(),
                      positive_count=int((measured > 0).sum()), negative_count=int((measured < 0).sum()),
                      zero_count=int((measured == 0).sum()))
    return result


def group_masks(batch, mask, torch):
    A, Q = batch["advantages"], batch["dt_q_estimates"]
    signs = {"positive": A > 0, "negative": A < 0, "zero": A == 0}
    values = {"Q0": Q == 0, "Q1": Q == 1, "Q_other": (Q != 0) & (Q != 1)}
    groups = {"all_policy_tokens": torch.ones_like(mask, dtype=torch.bool)}
    groups.update({"DT_A_" + name: selected for name, selected in signs.items()})
    groups.update({"saved_DT_" + name: selected for name, selected in values.items()})
    groups.update({f"saved_DT_{name}_A_{sign}": selected & signs[sign]
                   for name, selected in values.items() for sign in signs})
    row_returns = batch["token_level_rewards"].sum(dim=-1)
    for name, rows in {"row_return0": row_returns == 0, "row_return1": row_returns == 1,
                       "row_return_other": (row_returns != 0) & (row_returns != 1)}.items():
        groups[name] = rows[:, None].expand_as(mask)
    return groups, row_returns


def pg(core, config, batch, lp, A, mask):
    # The official implementation owns ratio, clipping, dual clipping and
    # aggregation. No hand-written surrogate or reweighted mask is used.
    loss, clip, ppo_kl, lower = core.compute_policy_loss(
        old_log_prob=batch["old_log_probs"], log_prob=lp, advantages=A,
        response_mask=mask, cliprange=config["clip_ratio"],
        cliprange_low=config["clip_ratio_low"], cliprange_high=config["clip_ratio_high"],
        clip_ratio_c=config["clip_ratio_c"], loss_agg_mode=config["loss_agg_mode"])
    return float(loss)


def equal_B4_means(rows, branches, group_names):
    result = {}
    for group in group_names:
        result[group] = {"B4_count": len(rows),
                         "policy_denominator_fraction_equal_B4_mean": sum(row["groups"][group]["original_denominator_fraction"] for row in rows) / len(rows),
                         "token_count_pooled_description": sum(row["groups"][group]["policy_token_count"] for row in rows),
                         "branches": {}}
        for branch in branches:
            first = rows[0]["groups"][group]["branches"][branch]["owner_denominator_contributions"]
            result[group]["branches"][branch] = {key: sum(row["groups"][group]["branches"][branch]["owner_denominator_contributions"][key] for row in rows) / len(rows) for key in first}
        result[group]["DT_minus_regularizers_only_extra_response"] = {
            key: sum(row["groups"][group]["DT_minus_regularizers_only_extra_response"]["owner_denominator_contributions"][key] for row in rows) / len(rows)
            for key in rows[0]["groups"][group]["DT_minus_regularizers_only_extra_response"]["owner_denominator_contributions"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--minibatch-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use the recorded native CPU environment with CUDA hidden.")
    if args.output.exists():
        raise FileExistsError("Preserve the first complete diagnostic result.")
    import torch
    from verl.protocol import DataProto
    from verl.trainer.ppo import core_algos as core
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("No CUDA or process-group initialization belongs to this analysis.")
    started, resource_before = time.time(), resources(torch)
    source_records = {"analyzer": identity(__file__), "helper": identity(Path(__file__).with_name("analyze_textcraft_native_adam.py")),
                      "protocol_load": source(DataProto.load_from_disk), "protocol_chunk": source(DataProto.chunk),
                      "policy_loss": source(core.compute_policy_loss), "agg_loss": source(core.agg_loss), "torch": identity(torch.__file__)}
    if source_records["helper"]["sha256"] != HELPER_SHA:
        raise ValueError("Use the already frozen original native analysis helper.")
    if source_records["protocol_load"]["sha256"] != PROTO_SHA or source_records["protocol_chunk"]["sha256"] != PROTO_SHA:
        raise ValueError("Use the recorded original DataProto owner.")
    if source_records["policy_loss"]["sha256"] != CORE_SHA or source_records["agg_loss"]["sha256"] != CORE_SHA:
        raise ValueError("Use the recorded original core owner.")
    inputs = {"minibatch": identity(args.minibatch_path)}
    frozen = {}
    for name, digest in (("completed.json", COMPLETED_SHA), ("native-owner-inspection.json", INSPECTION_SHA),
                         ("native-objective-effect.json", OBJECTIVE_SHA)):
        path = args.input_dir / name
        inputs[name] = identity(path)
        if inputs[name]["sha256"] != digest:
            raise ValueError("Original frozen source receipt changed: " + name)
        frozen[name] = json.loads(path.read_bytes())
    complete, config = frozen["completed.json"], frozen["native-owner-inspection.json"]["actor_config"]["actor"]
    if inputs["minibatch"]["sha256"] != MINIBATCH_SHA or complete["source_minibatch_sha256"] != MINIBATCH_SHA:
        raise ValueError("Use the original saved native global64 batch.")
    if (config["ppo_mini_batch_size"] != 64 or config["ppo_micro_batch_size_per_gpu"] != 4
            or config["policy_loss"]["loss_mode"] != "vanilla"):
        raise ValueError("This observation targets the original native global64/B4 vanilla objective.")
    saved = DataProto.load_from_disk(str(args.minibatch_path))
    if len(saved) != 64 or not saved.meta_info.get("multi_turn"):
        raise ValueError("Use the original multi-turn native64 carrier.")
    for field in ("responses", "advantages", "dt_token_advantages", "dt_q_estimates", "diagnostic_grpo_advantages",
                  "token_level_rewards", "old_log_probs", "ref_log_prob", "loss_mask"):
        if field not in saved.batch:
            raise ValueError("Missing original saved field: " + field)
    if any(value.device.type != "cpu" for value in saved.batch.values()):
        raise ValueError("Only saved CPU tensors are used.")
    original_ranks = saved.chunk(2)
    complete_branches = {row["branch"]: row for row in complete["branches"]}
    observations = {}
    for branch in BRANCHES:
        observations[branch] = {}
        for when in ("before", "after"):
            path = args.input_dir / f"{branch}-{when}-logprob.pkl"
            inputs[path.name] = identity(path)
            if inputs[path.name]["sha256"] != complete_branches[branch][when]["sha256"]:
                raise ValueError("Actual saved native LP/H file differs from completion receipt.")
            data = DataProto.load_from_disk(str(path))
            if len(data) != 64 or any(tensor.device.type != "cpu" for tensor in data.batch.values()):
                raise ValueError("Use unchanged native CPU observations for all64 rows.")
            observations[branch][when] = data.chunk(2)
    initial = {branch: {str(rank): {field: torch.equal(observations[branch]["before"][rank].batch[field], observations["dt"]["before"][rank].batch[field])
                                    for field in ("old_log_probs", "entropys")} for rank in range(2)} for branch in BRANCHES}
    rank_records, crosschecks = [], []
    with torch.no_grad():
        for rank, original in enumerate(original_ranks):
            micro = original.batch.split(4)
            if len(original) != 32 or len(micro) != 8:
                raise ValueError("Keep the original local32 -> continuous B4x8 partition.")
            rows = []
            for index, batch in enumerate(micro):
                width = batch["responses"].shape[-1]
                mask = batch["loss_mask"][:, -width:]
                if not torch.equal(batch["advantages"], batch["dt_token_advantages"]):
                    raise ValueError("Saved actor DT A is not the original credit field.")
                groups, row_returns = group_masks(batch, mask, torch)
                denominator = float(mask.sum())
                def reduce(value):
                    return float(core.agg_loss(loss_mat=value, loss_mask=mask, loss_agg_mode=config["loss_agg_mode"]))
                def contribution(value, selected):
                    return reduce(torch.where(selected, value, torch.zeros_like(value)))
                observed = {branch: {when: observations[branch][when][rank].batch[index*4:(index+1)*4]
                                     for when in ("before", "after")} for branch in BRANCHES}
                for branch in BRANCHES:
                    for when in ("before", "after"):
                        if any(observed[branch][when][field].shape != mask.shape for field in ("old_log_probs", "entropys")):
                            raise ValueError("Observation columns differ from the original saved actor mask.")
                row = {"rank": rank, "microbatch_index": index,
                       "global_rows": list(range(rank*32+index*4, rank*32+index*4+4)),
                       "traj_uid": [str(x) for x in original.non_tensor_batch["traj_uid"][index*4:(index+1)*4]],
                       "saved_training_row_returns": row_returns.tolist(), "original_policy_mask_denominator": denominator,
                       "groups": {}}
                for name, membership in groups.items():
                    selected = membership & (mask != 0)
                    count = int(selected.sum())
                    group = {"policy_token_count": count, "original_denominator_fraction": count/denominator if denominator else None,
                             "saved_DT_A": descriptive(batch["advantages"], selected, torch), "branches": {}}
                    for branch in BRANCHES:
                        before, after = observed[branch]["before"], observed[branch]["after"]
                        old_lp, new_lp = before["old_log_probs"], after["old_log_probs"]
                        LP_delta, H_delta = new_lp-old_lp, after["entropys"]-before["entropys"]
                        p_delta = new_lp.double().exp()-old_lp.double().exp()
                        scalars = {"LP_delta": contribution(LP_delta, selected), "probability_delta": contribution(p_delta, selected),
                                   "H_delta": contribution(H_delta, selected)}
                        for signal, field in (("dt", "advantages"), ("grpo", "diagnostic_grpo_advantages")):
                            # This is a diagnostic partition of saved A only.
                            # The original full actor mask/denominator stays fixed.
                            partition = torch.where(selected, batch[field], torch.zeros_like(batch[field]))
                            scalars[signal+"_pg_before"] = pg(core, config, batch, old_lp, partition, mask)
                            scalars[signal+"_pg_after"] = pg(core, config, batch, new_lp, partition, mask)
                            scalars[signal+"_pg_change"] = scalars[signal+"_pg_after"]-scalars[signal+"_pg_before"]
                        group["branches"][branch] = {"LP_delta_selected_description": descriptive(LP_delta, selected, torch),
                                                     "probability_delta_selected_description": descriptive(p_delta, selected, torch),
                                                     "owner_denominator_contributions": scalars}
                    dt_before, dt_after = observed["dt"]["before"]["old_log_probs"], observed["dt"]["after"]["old_log_probs"]
                    reg_before, reg_after = observed["regularizers_only"]["before"]["old_log_probs"], observed["regularizers_only"]["after"]["old_log_probs"]
                    extra_lp = (dt_after-dt_before)-(reg_after-reg_before)
                    extra_p = (dt_after.double().exp()-dt_before.double().exp())-(reg_after.double().exp()-reg_before.double().exp())
                    extra = {"LP_delta": contribution(extra_lp, selected), "probability_delta": contribution(extra_p, selected)}
                    for signal in ("dt", "grpo"):
                        extra[signal+"_pg_change"] = (group["branches"]["dt"]["owner_denominator_contributions"][signal+"_pg_change"]
                                                      -group["branches"]["regularizers_only"]["owner_denominator_contributions"][signal+"_pg_change"])
                    group["DT_minus_regularizers_only_extra_response"] = {
                        "LP_extra_selected_description": descriptive(extra_lp, selected, torch),
                        "probability_extra_selected_description": descriptive(extra_p, selected, torch),
                        "owner_denominator_contributions": extra}
                    row["groups"][name] = group
                native_original = frozen["native-objective-effect.json"]["branches"]
                for branch in BRANCHES:
                    previous = native_original[branch]["ranks"][rank]["microbatches"][index]
                    current = row["groups"]["all_policy_tokens"]["branches"][branch]["owner_denominator_contributions"]
                    crosschecks.append({"rank": rank, "microbatch_index": index, "branch": branch,
                                        "DT_pg_before_minus_existing_native_core": current["dt_pg_before"]-previous["outcomes"]["before"]["dt_pg"],
                                        "DT_pg_after_minus_existing_native_core": current["dt_pg_after"]-previous["outcomes"]["after"]["dt_pg"],
                                        "GRPO_pg_before_minus_existing_native_core": current["grpo_pg_before"]-previous["outcomes"]["before"]["grpo_pg"],
                                        "GRPO_pg_after_minus_existing_native_core": current["grpo_pg_after"]-previous["outcomes"]["after"]["grpo_pg"]})
                rows.append(row)
            rank_records.append({"rank": rank, "B4": rows, "native_equal_B4_sum_divided_by_8": equal_B4_means(rows, BRANCHES, groups)})
    all_rows = [row for rank in rank_records for row in rank["B4"]]
    combined = equal_B4_means(all_rows, BRANCHES, all_rows[0]["groups"])
    result = {
        "scope": __doc__, "sources": source_records, "inputs": inputs, "effective_original_actor_config": config,
        "batch_field_schema": {field: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)} for field, value in saved.batch.items()},
        "original_reduction": "Original DataProto.chunk(2): rank32 -> continuous B4x8 -> sum/8; final mean of two rank reductions. Every LP/H contribution and native PG partition retains the full original B4 mask and denominator.",
        "conditioned_statistic_scope": "Selected-description means/RMS are token-conditional descriptive quantities, separately labeled; they never replace the original B4 loss denominator. Probability deltas use FP64 exp of saved native LP, not a new probability prediction.",
        "G_definitions": {"saved_DT_Q": "Saved dt_q_estimates at original policy-token coordinates is the owner Q_hat=G_t field; no reward reconstruction or assignment is performed.",
                          "saved_row_return": "Original token_level_rewards.sum(-1), the official GRPO row-score reduction. It is reported separately from per-response G_t; no assumed equality.",
                          "A_sign": "Original advantages/dt_token_advantages sign; zero does not identify a correct/incorrect action."},
        "initial_LP_H_array_comparisons_across_branches": initial,
        "ranks": rank_records, "mean_of_two_native_rank_reductions": combined,
        "crosscheck_against_existing_official_core_objective": crosschecks,
        "interpretation_limits": [
            "This is one saved checkpoint25 actual Adam experiment; no new success rate or historical multi-step causality follows from its fixed-input LP/loss changes.",
            "Success rows can contain harmful intermediate actions. Positive/negative/zero A is a saved signal category, not an action-correctness oracle; GRPO is not an oracle either.",
            "Regularizers-only retains restored Adam moments and weight decay. DT-minus-control probability/LP or loss differences are matched response contrasts, not a linear Adam contribution decomposition.",
            "The existing native total-gradient difference projects down the DT frozen task objective to first order. Finite eval-based PG changes, gradient-difference rounding, curvature and native training/evaluation paths must remain distinct.",
            "Loss partitions zero advantages outside each diagnostic subset while retaining the original full mask. No training array, loss rule, config, reward, estimator, normalization, scale, or gate is changed.",
        ],
        "operations": {"model_initializations": 0, "model_forwards": 0, "sampling": 0, "DT": 0, "backward": 0, "optimizer_updates": 0},
        "runtime": {"started_unix": started, "completed_unix": time.time(), "resources_before": resource_before, "resources_after": resources(torch)},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    selected = {key: combined[key] for key in ("all_policy_tokens", "DT_A_positive", "DT_A_negative", "DT_A_zero")}
    print(json.dumps({"output": identity(args.output), "native_B4_summary": selected, "runtime": result["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
