"""Describe retained terminal credit from saved native TextCraft artifacts only.

No model, DT, gradient, reward, or PPO calculation is run by this analysis.
The coefficient weights below use the recorded native B4 loss denominators.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path


INPUTS = {
    "response": "native-minibatch-v4/response-credit-length-decomposition.json",
    "terminal": "terminal-event-readout-source-observation-20261006.json",
    "native_B4": "readout-quality-20261006/v2/native-b4-denominator-analysis.json",
    "mapping": "readout-quality-20261006/v2/original-readout-mapped-requests.json",
    "carrier": "native-minibatch-v4/native-minibatch-analysis.json",
}


def identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def summarize(rows: list[dict], world_size: int) -> dict:
    def total(field: str) -> float:
        return math.fsum(row[field] for row in rows)

    result = {
        "responses": len(rows),
        "unique_traj_uid": len({row["traj_uid"] for row in rows}),
        "source_policy_tokens": sum(row["L"] for row in rows),
        "nonzero_A_tokens": sum(row["A_nonzero"] for row in rows),
        "positive_A_tokens": sum(row["A_positive_count"] for row in rows),
        "negative_A_tokens": sum(row["A_negative_count"] for row in rows),
        "d_signed_sum": total("d_sum"),
        "d_L1": total("d_abs_sum"),
        "A_signed_sum": total("A_sum"),
        "A_L1": total("A_abs_sum"),
        "A_positive_mass": math.fsum(
            (row["A_abs_sum"] + row["A_sum"]) / 2 for row in rows
        ),
        "A_negative_abs_mass": math.fsum(
            (row["A_abs_sum"] - row["A_sum"]) / 2 for row in rows
        ),
        "sum_per_response_abs_Asum": math.fsum(abs(row["A_sum"]) for row in rows),
        "sum_per_response_abs_dsum": math.fsum(abs(row["d_sum"]) for row in rows),
        "mean_per_response_A_cancellation": total("A_cancellation") / len(rows),
        "mean_per_response_d_cancellation": total("d_cancellation") / len(rows),
    }
    result["A_token_abs_mean"] = result["A_L1"] / result["source_policy_tokens"]
    result["A_token_signed_mean"] = result["A_signed_sum"] / result["source_policy_tokens"]
    result["within_response_A_cancelled_mass"] = (
        result["A_L1"] - result["sum_per_response_abs_Asum"]
    )
    result["between_response_A_cancelled_mass"] = (
        result["sum_per_response_abs_Asum"] - abs(result["A_signed_sum"])
    )
    result["A_pooled_signed_over_abs"] = result["A_signed_sum"] / result["A_L1"]
    result["d_pooled_signed_over_abs"] = result["d_signed_sum"] / result["d_L1"]
    result["coefficient_original_B4_accumulation_DPmean"] = {
        "signed": math.fsum(
            row["A_sum"] / row["denominator"] / row["accumulation_divisor"] / world_size
            for row in rows
        ),
        "L1": math.fsum(
            row["A_abs_sum"] / row["denominator"] / row["accumulation_divisor"] / world_size
            for row in rows
        ),
        "source_token_fraction": math.fsum(
            row["L"] / row["denominator"] / row["accumulation_divisor"] / world_size
            for row in rows
        ),
    }
    return result


def analyze(data_root: Path) -> dict:
    paths = {name: data_root / relative for name, relative in INPUTS.items()}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    terminals = {
        (row["traj_uid"], row["source_step"]): row
        for row in data["terminal"]["rows"]
    }
    responses = {
        (row["traj_uid"], row["source_step"]): row
        for row in data["response"]["per_response_summaries"]
    }
    requests = collections.defaultdict(list)
    for row in data["mapping"]["requests"]:
        requests[(row["traj_uid"], row["env_step"])].append(row)
    if set(terminals) != set(responses) or set(responses) != set(requests):
        raise ValueError("Saved response identities do not match the terminal/source mapping")

    batches = {
        (batch["rank"], batch["microbatch_index"]): batch
        for batch in data["native_B4"]["B4"]
    }
    actor_rows = {row["global_row"]: row for row in data["native_B4"]["rows"]}
    ranks = sorted({rank for rank, _ in batches})
    if ranks != [0, 1] or len(batches) != 16:
        raise ValueError("This saved native carrier is not the recorded two-rank 16-B4 artifact")
    if any(batch["native_accumulation_divisor"] != 8 for batch in batches.values()):
        raise ValueError("Recorded native accumulation divisor differs from this artifact's /8")

    joined = []
    replica_ranges = []
    partial_slices = 0
    for key, copies in requests.items():
        response = responses[key]
        terminal = terminals[key]
        slices = [saved for copy in copies for saved in copy["saved_actor_slices"]]
        coordinates = {
            (saved["actor_row"], saved["actor_response_start"], saved["retained_tokens"])
            for saved in slices
        }
        if len(coordinates) != 1:
            raise ValueError(f"Source identity {key} does not identify one original actor slice")
        partial_slices += sum(not saved["full_response_retained"] for saved in slices)
        if any(saved["retained_tokens"] != response["L"] for saved in slices):
            raise ValueError(f"Source length mismatch for {key}")
        signed_values = [saved["saved_A_summary"]["signed_sum"] for saved in slices]
        absolute_values = [saved["saved_A_summary"]["abs_sum"] for saved in slices]
        signed_range = max(signed_values) - min(signed_values)
        absolute_range = max(absolute_values) - min(absolute_values)
        replica_ranges.append({
            "traj_uid": key[0], "source_step": key[1], "transport_slots": len(copies),
            "A_signed_sum_range": signed_range, "A_L1_range": absolute_range,
        })
        # The frozen artifacts recorded identical A summaries for transport copies.
        # Do not silently average a different carrier's replicas.
        if signed_range != 0 or absolute_range != 0:
            raise ValueError(f"Transport replicas differ in saved A for {key}")
        saved = slices[0]
        actor_row = actor_rows[saved["actor_row"]]
        if actor_row["traj_uid"] != key[0]:
            raise ValueError(f"Original actor UID mismatch for {key}")
        if absolute_values[0] != response["A_abs_sum"] or signed_values[0] != response["A_sum"]:
            raise ValueError(f"Source saved-A summary mismatch for {key}")
        batch = batches[(actor_row["rank"], actor_row["microbatch_index"])]
        joined.append({
            **response,
            "is_final_success_response": terminal["is_final_success_response"],
            "actor_row": saved["actor_row"],
            "actor_response_start": saved["actor_response_start"],
            "actor_rank": actor_row["rank"],
            "actor_microbatch_index": actor_row["microbatch_index"],
            "denominator": batch["original_mask_denominator"],
            "accumulation_divisor": batch["native_accumulation_divisor"],
            "A_nonzero": saved["saved_A_summary"]["nonzero_count"],
            "A_positive_count": saved["saved_A_summary"]["positive_count"],
            "A_negative_count": saved["saved_A_summary"]["negative_count"],
        })

    terminal_rows = [row for row in joined if row["is_final_success_response"]]
    earlier_rows = [row for row in joined if not row["is_final_success_response"]]
    stages = {
        "terminal": summarize(terminal_rows, len(ranks)),
        "earlier": summarize(earlier_rows, len(ranks)),
        "all_G1": summarize(joined, len(ranks)),
    }
    terminal = stages["terminal"]
    overall = stages["all_G1"]
    carrier = data["carrier"]["native_minibatch"]
    carrier_mass = carrier["original_token_id_mass"]["finite_mass_totals"]["dt_actor_advantages"]
    return {
        "scope": "Saved checkpoint25 native64 TextCraft G1 response credit and original actor mask support; descriptive coefficients only",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {"analyzer": identity(Path(__file__)), **{name: identity(path) for name, path in paths.items()}},
        "population": {
            "original_transport_slots": len(data["mapping"]["requests"]),
            "unique_G1_responses": len(joined),
            "G1_traj_uid": len({row["traj_uid"] for row in joined}),
            "terminal_responses": len(terminal_rows),
            "earlier_responses": len(earlier_rows),
            "all_carrier_rows": len(actor_rows),
            "all_carrier_policy_mask_tokens": carrier["mask"]["action_token_count"],
            "source_policy_tokens_include": "Every generated reasoning, tool-call, and final token; not an environment-command-only classification",
            "primary_replica": "First original transport occurrence; all saved A signed/L1 summary replica ranges are separately retained",
        },
        "native_weighting": {
            "original_loss_mask": carrier["mask"],
            "world_size": len(ranks),
            "per_rank_original_B4_count": {str(rank): sum(r == rank for r, _ in batches) for rank in ranks},
            "description": "For each saved response: A mass / its recorded original B4 mask denominator / recorded native accumulation divisor 8; then mean over two ranks. This is a coefficient scalar, not a gradient or update partition.",
            "B4_denominators": [{
                "rank": batch["rank"], "microbatch_index": batch["microbatch_index"],
                "global_row_indices": batch["global_row_indices"],
                "original_mask_denominator": batch["original_mask_denominator"],
                "native_accumulation_divisor": batch["native_accumulation_divisor"],
            } for batch in data["native_B4"]["B4"]],
        },
        "stages": stages,
        "terminal_fractions": {
            "G1_policy_tokens": terminal["source_policy_tokens"] / overall["source_policy_tokens"],
            "all_carrier_policy_mask_tokens": terminal["source_policy_tokens"] / carrier["mask"]["action_token_count"],
            "G1_A_L1": terminal["A_L1"] / overall["A_L1"],
            "G1_A_signed_sum": terminal["A_signed_sum"] / overall["A_signed_sum"],
            "original_B4_weighted_G1_L1": terminal["coefficient_original_B4_accumulation_DPmean"]["L1"] / overall["coefficient_original_B4_accumulation_DPmean"]["L1"],
            "original_B4_weighted_G1_signed": terminal["coefficient_original_B4_accumulation_DPmean"]["signed"] / overall["coefficient_original_B4_accumulation_DPmean"]["signed"],
        },
        "mask_and_scatter_closure": {
            "partial_response_slices": partial_slices,
            "mapped_source_A_L1": overall["A_L1"],
            "original_carrier_masked_A_L1": carrier_mass["abs_mass"],
            "A_L1_difference": overall["A_L1"] - carrier_mass["abs_mass"],
            "mapped_source_A_signed_sum": overall["A_signed_sum"],
            "original_carrier_masked_A_signed_sum": carrier_mass["signed_mass"],
            "A_signed_sum_difference": overall["A_signed_sum"] - carrier_mass["signed_mass"],
            "mapped_nonzero_A_tokens": overall["nonzero_A_tokens"],
            "original_carrier_masked_nonzero_A_tokens": carrier_mass["positive_count"] + carrier_mass["negative_count"],
            "original_carrier_A_outside_mask_nonzero": carrier["outside_actor_mask"]["dt_actor_advantages"]["nonzero_count"],
            "torch_equal_saved_dt_and_actor_advantages": carrier["dt_actor_binding"]["torch_equal_saved_dt_token_advantages"],
            "max_replica_A_signed_sum_range": max(row["A_signed_sum_range"] for row in replica_ranges),
            "max_replica_A_L1_range": max(row["A_L1_range"] for row in replica_ranges),
            "source_scope": "Closure concerns saved full response source spans through the saved actor carrier. It does not reconstruct an unrecorded pre-extraction full-input attribution vector.",
        },
        "per_response_original_actor_mapping": joined,
        "transport_replica_ranges": replica_ranges,
        "limitations": [
            "G1-only response population: 186 responses of 21 successful trajectories. It is not a representative population of failed trajectories or historical training.",
            "A L1 shares and original B4 coefficient weights are not gradient contribution shares, parameter-update shares, or an explanation of the 1/211 gradient norm ratio.",
            "Signed coefficient cancellation within/between responses is not Jacobian-weighted gradient-vector cancellation.",
            "Terminal identity is inherited from the source-bound official done-stop observation; no environment counterfactual or new reward is evaluated.",
            "d statistics are inverse-composed from saved FP32 A in the frozen source artifact, not a new DT run or a recovery of its pre-storage vector to arbitrary precision.",
            "Complete saved-source-to-mask mass closure does not prove exact single-token counterfactual values, reward readout accuracy, overall DT faithfulness, or a repair strategy.",
            "No fixed 1/L causal rule is inferred from attribution conservation or these response support statistics.",
        ],
        "operations": {"new_model_forward": 0, "new_DT": 0, "new_gradient": 0, "optimizer_step": 0, "environment_step": 0, "training_or_method_changes": 0},
    }


def main() -> None:
    default_root = Path(__file__).resolve().parent / "textcraft-degradation-20261005"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_root / "terminal-credit-support-20261006.json")
    args = parser.parse_args()
    result = analyze(args.data_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": identity(args.output), "terminal_fractions": result["terminal_fractions"], "mask_and_scatter_closure": result["mask_and_scatter_closure"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
