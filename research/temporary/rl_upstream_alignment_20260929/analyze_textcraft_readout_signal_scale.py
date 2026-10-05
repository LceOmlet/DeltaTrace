"""Describe already-saved native64 readout and actor credit on CPU only.

Joint EOS endpoint scores and saved per-token credit remain separate. No model,
new credit calculation, optimizer, calibration or numerical acceptance test.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import psutil
import torch
from verl.protocol import DataProto
import analyze_textcraft_native_minibatch as owner_stats


def stats(values):
    values = list(values)
    present = [x for x in values if x is not None]
    result = owner_stats.summary(torch.as_tensor(present, dtype=torch.float64), torch)
    result["missing_count"] = len(values) - len(present)
    return result


def pearson(rows, left, right):
    pairs = [(r[left], r[right]) for r in rows
             if r[left] is not None and r[right] is not None
             and math.isfinite(r[left]) and math.isfinite(r[right])]
    if len(pairs) < 2:
        return {"count": len(pairs), "pearson": None}
    values = torch.tensor(pairs, dtype=torch.float64).T
    if bool((values.std(dim=1) == 0).any()):
        return {"count": len(pairs), "pearson": None, "reason": "constant variable"}
    return {"count": len(pairs), "pearson": float(torch.corrcoef(values)[0, 1])}


def cancellation(values):
    signed, absolute = float(values.sum()), float(values.abs().sum())
    return {"signed_sum": signed, "absolute_sum": absolute,
            "positive_mass": float(values[values > 0].sum()),
            "negative_absolute_mass": float(-values[values < 0].sum()),
            "positive_tokens": int((values > 0).sum()),
            "negative_tokens": int((values < 0).sum()),
            "zero_tokens": int((values == 0).sum()),
            "signed_over_absolute": signed / absolute if absolute else None,
            "absolute_signed_over_absolute": abs(signed) / absolute if absolute else None,
            "cancellation_fraction": 1 - abs(signed) / absolute if absolute else None}


def response_summary(rows):
    return {"responses": len(rows), "uids": len({r["uid"] for r in rows}),
            "tokens": sum(r["length"] for r in rows),
            "length": stats([r["length"] for r in rows]),
            "factual_p1": stats([r["factual_p1"] for r in rows]),
            "full_response_EOS_p1": stats([r["full_EOS_p1"] for r in rows]),
            "factual_minus_full_EOS_p1": stats([r["probability_difference"] for r in rows]),
            "joint_root": stats([r["root"] for r in rows]),
            "per_response_mean_abs_d_from_saved_A": stats([r["d_abs_mean"] for r in rows]),
            "per_response_median_abs_d_from_saved_A": stats([r["d_abs_median"] for r in rows]),
            "per_response_mean_abs_A": stats([r["A_abs_mean"] for r in rows]),
            "per_response_median_abs_A": stats([r["A_abs_median"] for r in rows]),
            "per_response_A_cancellation_fraction": stats([r["A_cancellation"] for r in rows if r["A_cancellation"] is not None]),
            "per_response_d_cancellation_fraction": stats([r["d_cancellation"] for r in rows if r["d_cancellation"] is not None])}


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    torch.set_num_threads(1)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    mapping_path = out / "native-minibatch-readout-mapping.json"
    mapping = json.loads(mapping_path.read_bytes())
    request_path = Path(mapping["mapped_requests_artifact"]["path"])
    actor_path = Path(mapping["inputs"]["actor_minibatch"]["path"])
    for path, expected in ((request_path, mapping["mapped_requests_artifact"]),
                           (actor_path, mapping["inputs"]["actor_minibatch"])):
        if owner_stats.identity(path) != expected:
            raise ValueError("Saved input identity changed: " + str(path))
    slots = json.loads(request_path.read_bytes())["requests"]
    actor_data = DataProto.load_from_disk(str(actor_path))
    if any(x.device.type != "cpu" for x in actor_data.batch.values()):
        raise ValueError("Original saved actor carrier is not CPU only")
    groups = defaultdict(list)
    for slot in slots:
        groups[(slot["traj_uid"], slot["env_step"])].append(slot)
    rows, token_A, token_d = [], [], []
    duplicate_ranges = []
    inverse_signed_gaps = []
    for (uid, step), copies in groups.items():
        first = copies[0]
        if any(len(slot["saved_actor_slices"]) != 1 for slot in copies):
            raise ValueError("Expected the already-verified single original slice per response")
        source = first["saved_actor_slices"][0]
        actor_row, start, length = (source[k] for k in
                                  ("actor_row", "actor_response_start", "retained_tokens"))
        for slot in copies:
            other = slot["saved_actor_slices"][0]
            if (other["actor_row"], other["actor_response_start"], other["retained_tokens"]) != (actor_row, start, length):
                raise ValueError("Duplicate native transport slots have different saved credit slices")
        if length != first["source_end"] - first["source_start"] or not source["full_response_retained"]:
            raise ValueError("This existing mapping no longer has its recorded complete response coverage")
        g = source["actual_G"]
        if g != 1:
            raise ValueError("The recorded nonzero TextCraft readout set is not G1")
        A = actor_data.batch["dt_token_advantages"][actor_row, start:start + length].double()
        if not torch.equal(A, actor_data.batch["advantages"][actor_row, start:start + length].double()):
            raise ValueError("Saved actor and DT advantage fields differ")
        # Original actor/analyzer selects only the response-width tail of the
        # full input loss_mask before applying response-relative slice offsets.
        response_width = actor_data.batch["responses"].shape[-1]
        mask = actor_data.batch["loss_mask"][actor_row, -response_width:][start:start + length]
        if not bool((mask != 0).all()):
            raise ValueError("Original mapped generated response includes masked observation/padding")
        # Diagnostic inversion of previously stored FP32 advantage. This is not
        # the original raw finite d and is never used to produce training credit.
        d = -torch.log1p(-A / g)
        ds, As = cancellation(d), cancellation(A)
        factual = sum(x["factual_probability"] for x in copies) / len(copies)
        reference = sum(x["whole_response_EOS_probability"] for x in copies) / len(copies)
        root = sum(x["trace"]["root_effect"] for x in copies) / len(copies)
        record = {"uid": uid, "step": step, "length": length, "transport_slots": len(copies),
                  "factual_p1": factual, "full_EOS_p1": reference,
                  "probability_difference": factual - reference, "root": root, "root_abs": abs(root),
                  "d_abs_mean": float(d.abs().mean()), "d_abs_median": float(d.abs().median()),
                  "d_absolute_sum": ds["absolute_sum"], "d_signed_sum": ds["signed_sum"],
                  "d_cancellation": ds["cancellation_fraction"],
                  "A_abs_mean": float(A.abs().mean()), "A_abs_median": float(A.abs().median()),
                  "A_absolute_sum": As["absolute_sum"], "A_signed_sum": As["signed_sum"],
                  "A_cancellation": As["cancellation_fraction"],
                  "d_masses": ds, "A_masses": As}
        rows.append(record)
        token_A.append(A)
        token_d.append(d)
        inverse_signed_gaps.append(source["inferred_d_sum_minus_logged_signed_sum"])
        if len(copies) > 1:
            duplicate_ranges.append({"uid": uid, "step": step, "slots": len(copies),
                "factual_p1_range": max(x["factual_probability"] for x in copies) - min(x["factual_probability"] for x in copies),
                "full_EOS_p1_range": max(x["whole_response_EOS_probability"] for x in copies) - min(x["whole_response_EOS_probability"] for x in copies),
                "joint_root_range": max(x["trace"]["root_effect"] for x in copies) - min(x["trace"]["root_effect"] for x in copies)})
    per_uid = defaultdict(list)
    per_step = defaultdict(list)
    for row in rows:
        per_uid[row["uid"]].append(row)
        per_step[row["step"]].append(row)
    uid_balanced = []
    for uid, uid_rows in per_uid.items():
        uid_balanced.append({"uid": uid, "responses": len(uid_rows),
            "tokens": sum(r["length"] for r in uid_rows),
            **{key: sum(r[key] for r in uid_rows) / len(uid_rows) for key in
               ("factual_p1", "full_EOS_p1", "probability_difference", "root",
                "d_abs_mean", "A_abs_mean", "d_signed_sum", "d_absolute_sum",
                "A_signed_sum", "A_absolute_sum")}})
    balanced_keys = ("factual_p1", "full_EOS_p1", "probability_difference", "root", "d_abs_mean", "A_abs_mean")
    A_all, d_all = torch.cat(token_A), torch.cat(token_d)
    output_cap = json.loads((out / "launch.json").read_bytes())["options"]["data.max_response_length"]
    quartiles = torch.quantile(torch.tensor([r["length"] for r in rows], dtype=torch.float64),
                              torch.tensor([.25, .5, .75], dtype=torch.float64)).tolist()
    lower = [r for r in rows if r["length"] <= quartiles[0]]
    upper = [r for r in rows if r["length"] >= quartiles[2]]
    result = {"scope": __doc__, "created_unix": time.time(),
        "sources": {"script": owner_stats.identity(__file__), "mapping": owner_stats.identity(mapping_path),
                    "mapped_requests": owner_stats.identity(request_path), "original_actor_carrier": owner_stats.identity(actor_path),
                    "native_log": mapping["inputs"]["diagnostic_log"],
                    "original_protocol_load": owner_stats.source_identity(DataProto.load_from_disk),
                    "descriptive_summary": owner_stats.source_identity(owner_stats.summary),
                    "original_preparation_and_query_sources": mapping["sources"]},
        "coverage": {"native64_UIDs": len(actor_data), "G1_UIDs": len(per_uid), "G0_UIDs_not_scored": len(actor_data)-len(per_uid),
                     "logged_transport_slots": len(slots), "unique_responses": len(rows),
                     "unique_response_generated_tokens": A_all.numel(), "partial_response_slices": 0,
                     "transport_duplicates": len(slots)-len(rows), "duplicate_score_ranges": duplicate_ranges},
        "response_weighted": response_summary(rows),
        "UID_balanced": {"scope": "Average scores across unique responses within each successful UID, then weight each of the21 UIDs equally; duplicate endpoint evaluations averaged within one identity only",
                         "uids": len(per_uid), **{key: stats([u[key] for u in uid_balanced]) for key in balanced_keys}},
        "by_source_step": {str(step): response_summary(step_rows) for step, step_rows in sorted(per_step.items())},
        "per_UID": uid_balanced,
        "token_weighted": {"A": owner_stats.summary(A_all, torch), "d_from_saved_FP32_A": owner_stats.summary(d_all, torch),
                           "A_cancellation": cancellation(A_all), "d_cancellation": cancellation(d_all)},
        "per_response_signed_and_absolute": {key: stats([r[key] for r in rows]) for key in
            ("d_signed_sum", "d_absolute_sum", "A_signed_sum", "A_absolute_sum", "d_cancellation", "A_cancellation")},
        "length_associations": {"scope": "Descriptive Pearson correlations across186 unique responses; repeated responses from21 UIDs are not independent. No causal or significance claim",
            "length_vs": {key: pearson(rows, "length", key) for key in
                ("d_abs_mean", "d_abs_median", "d_absolute_sum", "A_abs_mean", "A_abs_median", "A_absolute_sum", "root_abs", "d_cancellation", "A_cancellation")},
            "observed_length_quartiles": quartiles,
            "lower_observed_length_quartile_including_ties": response_summary(lower),
            "upper_observed_length_quartile_including_ties": response_summary(upper),
            "at_original_output_cap": {"original_max_tokens": output_cap,
                "equal_cap": response_summary([r for r in rows if r["length"] == output_cap]),
                "below_cap": response_summary([r for r in rows if r["length"] < output_cap])}},
        "d_inverse_vs_original_finite_signed_sum": stats(inverse_signed_gaps),
        "limits": ["This existing readout logs G1 only;43 failed UIDs have no native readout probabilities here. The separate64 first-prefix native probe is needed to describe their probabilities.",
            "d here is diagnostic -log1p(-saved_FP32_A/G); it is not the original raw finite vector. A itself is the exact original saved array.",
            "Joint root is the whole-response EOS endpoint log-probability difference, not a single-token deletion effect or a test of token accuracy.",
            "Positive/negative cancellation and length associations do not prove a coding error or the cause of late task degradation.",
            "UID balancing changes descriptive weighting only; it does not normalize, alter or replace any training credit.",
            "No model forward, DT call, optimizer step, numerical pass gate, correction or production modification occurred."],
        "runtime": {"elapsed_seconds": time.time()-started, "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss,
                    "pss_after_bytes": process.memory_full_info().pss,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "distributed_initialized": torch.distributed.is_initialized()}}
    path = out / "textcraft-readout-signal-scale.json"
    path.write_text(json.dumps(owner_stats.json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": owner_stats.identity(path), "coverage": {k: result["coverage"][k] for k in
        ("logged_transport_slots", "unique_responses", "G1_UIDs", "unique_response_generated_tokens")},
        "UID_balanced_mean": {key: result["UID_balanced"][key]["mean"] for key in balanced_keys},
        "length_vs_mean_abs_d": result["length_associations"]["length_vs"]["d_abs_mean"],
        "length_vs_mean_abs_A": result["length_associations"]["length_vs"]["A_abs_mean"],
        "token_A_abs_mean": result["token_weighted"]["A"]["abs_mean"],
        "A_cancellation": result["token_weighted"]["A_cancellation"], "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
