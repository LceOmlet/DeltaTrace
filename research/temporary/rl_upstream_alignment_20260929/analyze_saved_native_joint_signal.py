"""Describe the already saved full-pair native DT scalar ledger (stdlib only).

This reads the frozen conditional-primitives v1 raw receipts.  It does not
contract tensors, call a model/finite rule, define a tolerance, or estimate
individual token-deletion accuracy.  The 26 first-response transport rows are
kept, with their 21 UID replica means reported separately.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


AUDIT = Path(__file__).resolve().parent
DEGRADATION = AUDIT / "textcraft-degradation-20261005"
DEFAULT_INPUT = DEGRADATION / "conditional-primitives-20261006" / "v1"
DEFAULT_OUTPUT = DEGRADATION / "author-cumulative-deletion-20261006" / "native-joint-signal-ledger.json"
RAW_SHA256 = {
    0: "9aa79af1470698cf265224a7eff6758b58df100f33e93361422c5ba90da2826d",
    1: "db400a134bfd1cf1063dbddfc5e0fe365be51ca22b1f64c3247e31a4a47c7e1f",
}
RUNNER_SHA256 = "c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1"
SCALARS = ("root", "seed", "final_signed", "policy_credit_signed_sum",
           "seed_minus_root", "final_minus_seed", "final_minus_root",
           "final_minus_policy_sum", "layer0_input_minus_final")


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": path.as_posix(), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def describe(values):
    values = list(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values),
            "abs_mean": statistics.fmean(abs(v) for v in values),
            "rms": math.sqrt(statistics.fmean(v * v for v in values)),
            "min": min(values), "max": max(values),
            "max_abs": max(abs(v) for v in values),
            "positive": sum(v > 0 for v in values),
            "negative": sum(v < 0 for v in values),
            "zero": sum(v == 0 for v in values)}


def one(records, phase):
    found = [record for record in records if record["phase"] == phase]
    if len(found) != 1:
        raise ValueError(f"Expected one original {phase}, found {len(found)}")
    return found[0]


def split_original_layer_records(records):
    """Parse original call order; do not compute a replacement finite effect."""
    inputs, outputs = {}, defaultdict(list)
    for record in records:
        layer = record["current_layer_index"]
        if record["phase"] == "layer_input_effect":
            if layer in inputs:
                raise ValueError(f"Repeated layer input record {layer}")
            inputs[layer] = record
        elif record["phase"] == "layer_root_or_replay_output_effect_shared_line":
            outputs[layer].append(record)
    if set(inputs) != set(range(32)) or set(outputs) != set(range(32)):
        raise ValueError("Original 32-layer scalar ledger is incomplete")
    for layer in range(32):
        if len(outputs[layer]) != 2:
            raise ValueError(f"Layer {layer} lacks original root/replay pair")
        # Original runner line 317 evaluates root_output_effect then replay.
        if not (outputs[layer][0]["call_index"] < outputs[layer][1]["call_index"]
                < inputs[layer]["call_index"]):
            raise ValueError(f"Original output/replay/input order differs at {layer}")
    return inputs, outputs


def scalar_summary(rows):
    return {name: describe(row[name] for row in rows) for name in SCALARS}


def layer_summary(rows):
    result = []
    for layer in range(32):
        result.append({
            "decoder_index": layer,
            "input": describe(row["layer_input"][layer] for row in rows),
            "root_output": describe(row["layer_root_output"][layer] for row in rows),
            "replay_output": describe(row["layer_replay_output"][layer] for row in rows),
            "input_minus_next_boundary": describe(
                row["layer_input"][layer] -
                (row["seed"] if layer == 31 else row["layer_input"][layer + 1])
                for row in rows),
            "root_minus_replay": describe(
                row["layer_root_output"][layer] - row["layer_replay_output"][layer]
                for row in rows),
        })
    return result


def first_response_replica_pool(rows):
    pools = defaultdict(list)
    for row in rows:
        if row["identity"]["source_step"] == 0:
            pools[(row["identity"]["traj_uid"], 0)].append(row)
    aggregate, descriptions = [], []
    for (uid, step), replicas in pools.items():
        means = {name: statistics.fmean(row[name] for row in replicas) for name in SCALARS}
        for name in ("layer_input", "layer_root_output", "layer_replay_output"):
            means[name] = [statistics.fmean(row[name][layer] for row in replicas)
                           for layer in range(32)]
        aggregate.append(means)
        descriptions.append({
            "traj_uid": uid, "source_step": step, "replica_count": len(replicas),
            "transport_ids": [row["identity"]["transport_id"] for row in replicas],
            "scalar_replica_means": {name: means[name] for name in SCALARS},
            "scalar_replica_ranges": {
                name: max(row[name] for row in replicas) - min(row[name] for row in replicas)
                for name in SCALARS},
            "layer_input_max_replica_range": max(
                max(row["layer_input"][layer] for row in replicas) -
                min(row["layer_input"][layer] for row in replicas)
                for layer in range(32)),
        })
    return aggregate, descriptions


def build(input_directory):
    sources, ranks, groups, rows = {}, [], [], []
    dtypes, shape_metadata, original_effect_sites = {}, {}, {}
    for rank in (0, 1):
        path = input_directory / f"rank{rank}-readout.json"
        source = identity(path)
        if source["sha256"] != RAW_SHA256[rank]:
            raise ValueError(f"Frozen raw SHA differs: {path}")
        sources[f"rank{rank}_raw"] = source
        raw = json.loads(path.read_bytes())
        if raw["rank"] != rank or raw["sources"]["runner"]["sha256"] != RUNNER_SHA256:
            raise ValueError("Raw rank or original runner identity differs")
        group_metadata = {group["original_owner_batch_index"]: group for group in raw["groups"]}
        full = [case for case in raw["cases"] if case["variant"] == "full_response_eos"]
        ranks.append({"rank": rank, "checkpoint": raw["checkpoint"],
                      "original_sources": raw["sources"],
                      "input_sha256": raw["input_sha256"],
                      "model_dtype": raw["model_dtype"],
                      "native_fla_fp16": raw["native_fla_fp16"],
                      "native_forward_calls": raw["native_forward_calls"],
                      "finite_trace_calls": raw["finite_trace_calls"],
                      "finite_seed_calls": raw["finite_seed_calls"],
                      "backward_calls": raw["backward_calls"],
                      "optimizer_steps": raw["optimizer_steps"],
                      "scheduler_steps": raw["scheduler_steps"],
                      "full_B4_cases_read": len(full)})
        for case in full:
            batch_index = case["original_owner_batch_index"]
            detail = case["original_finite_detail"]
            ledger = case["actual_coefficient_metadata"]["original_finite_effect_ledger"]
            if ledger["diagnostic_error"] is not None:
                raise ValueError(f"Original observation error: rank {rank}, batch {batch_index}")
            if ledger["owner_token_effect"]["sha256"] != RUNNER_SHA256:
                raise ValueError("Original token contraction source differs")
            records = ledger["records"]
            seed, final = one(records, "seed_effect"), one(records, "final_signed")
            inputs, outputs = split_original_layer_records(records)
            groups.append({"rank": rank, "original_owner_batch_index": batch_index,
                           "case_variant": case["variant"],
                           "observed_prefix_length": case["observed_prefix_length"],
                           "original_sequence_length": case["original_sequence_length"],
                           "original_paired_owner_abi": group_metadata[batch_index]["original_paired_owner_abi"],
                           "selection": case["selection"],
                           "ledger_call_count": ledger["call_count"],
                           "ledger_phase_counts": dict(Counter(record["phase"] for record in records)),
                           "compiled_seed_logprob_effect_minus_root": detail["compiled_seed_logprob_effect_minus_root"],
                           "conservation_verified_original_diagnostic": detail["conservation_verified"]})
            for record in records:
                name = record["phase"]
                dtype = {key: record[key] for key in
                         ("m_dtype", "x_dtype", "token_effect_dtype", "per_sample_sums_dtype")}
                dtypes[json.dumps(dtype, sort_keys=True)] = dtype
                shapes = {key: record[key] for key in
                          ("m_shape", "x_shape", "token_effect_shape", "per_sample_sums_shape")}
                shape_metadata[json.dumps(shapes, sort_keys=True)] = shapes
                site = record["attribute_call_site"]
                original_effect_sites[f"{name}:{site['line']}"] = site
            for slot, sample in enumerate(case["samples"]):
                saved = detail["per_sample"][slot]
                row = {"identity": {
                    "transport_id": f"rank{rank}/B4{batch_index}/slot{slot}",
                    "rank": rank, "original_owner_batch_index": batch_index, "slot": slot,
                    **{key: sample[key] for key in
                       ("traj_uid", "source_step", "original_request_index", "source_start", "source_end",
                        "context_tokens", "compute_tokens", "query_tokens", "observed_return")}},
                    "root": saved["root_effect"],
                    "reference_target_logp": saved["reference_target_logp"],
                    "factual_target_logp": saved["factual_target_logp"],
                    "seed": seed["per_sample_sums"][slot],
                    "final_signed": final["per_sample_sums"][slot],
                    "policy_credit_signed_sum": saved["policy_credit_signed_sum"],
                    "layer_input": [inputs[layer]["per_sample_sums"][slot] for layer in range(32)],
                    "layer_root_output": [outputs[layer][0]["per_sample_sums"][slot] for layer in range(32)],
                    "layer_replay_output": [outputs[layer][1]["per_sample_sums"][slot] for layer in range(32)]}
                row.update(seed_minus_root=row["seed"]-row["root"],
                           final_minus_seed=row["final_signed"]-row["seed"],
                           final_minus_root=row["final_signed"]-row["root"],
                           final_minus_policy_sum=row["final_signed"]-row["policy_credit_signed_sum"],
                           layer0_input_minus_final=row["layer_input"][0]-row["final_signed"])
                rows.append(row)
    first_rows = [row for row in rows if row["identity"]["source_step"] == 0]
    means, replica_pool = first_response_replica_pool(rows)
    return {"scope": __doc__, "raw_sources": sources,
            "actual_original_rank_sources_and_calls": ranks,
            "coverage": {"original_full_B4_cases": len(groups), "all_full_transport_rows": len(rows),
                         "first_response_transport_rows": len(first_rows),
                         "first_response_unique_UIDs": len(means),
                         "duplicated_first_response_UIDs": sum(item["replica_count"] > 1 for item in replica_pool),
                         "first_response_replica_histogram": dict(Counter(item["replica_count"] for item in replica_pool)),
                         "decoder_boundary_indices": list(range(32)),
                         "array_index_semantics": "layer arrays index the actual decoder 0..31; root/replay pair follows the original left-to-right call order"},
            "same_pair_first_response_unique_UID_summary": scalar_summary(means),
            "same_pair_first_response_transport_summary": scalar_summary(first_rows),
            "same_pair_all56_transport_summary": scalar_summary(rows),
            "same_pair_first_response_unique_UID_layer_summary": layer_summary(means),
            "same_pair_all56_transport_layer_summary": layer_summary(rows),
            "first_response_replica_pool": replica_pool,
            "original_full_B4_layouts": groups, "full_transport_chains": rows,
            "actual_original_dtype_sets": list(dtypes.values()),
            "actual_original_shape_sets": list(shape_metadata.values()),
            "original_contraction_source_sites": original_effect_sites,
            "operations_by_this_analysis": {"model_calls": 0, "tensor_contractions": 0,
                "new_DT_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
                "scheduler_steps": 0, "production_changes": 0, "new_tolerances": 0},
            "limits": [
                "The ledger is the original full-response EOS/factual pair. No single-token counterfactual is constructed or certified here.",
                "The 21 UID are successful first responses, with 26 correlated transport rows. All 56 B4 rows additionally include their original later-response partners.",
                "Root/seed/summed-credit agreement is not an FA/FLA official tolerance assertion, token-credit accuracy result, or a new gate.",
                "Per-layer scalar increments are observations of the same joint pair, not causal shares of weak PG or independent-deletion errors.",
                "GRPO group-standardized advantage and raw DT log-prob/return credit have different units. No amplitude multiplier or normalization is inferred.",
                "This ledger does not cover all186 formal responses/all generated tokens or explain the entire historical 1/211 gradient comparison.",
                "No raw tensor or coefficient was loaded or recomputed; the original per-sample JSON scalars and replica arithmetic are the only numerical inputs.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    report = build(args.source_directory)
    report["analysis_source"] = identity(__file__)
    report["analysis_wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "saved_original_joint_ledger_described",
                      "output": identity(args.output), "coverage": report["coverage"],
                      "unique_first_response": report["same_pair_first_response_unique_UID_summary"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
