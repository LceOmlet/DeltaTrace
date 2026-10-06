"""Describe saved response lengths and credit mass, without model or method changes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parent / "textcraft-degradation-20261005"


def source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    values = [float(v) for v in values]
    finite = sorted(v for v in values if math.isfinite(v))
    def quantile(p):
        if not finite:
            return None
        f = p * (len(finite) - 1)
        lo, hi = math.floor(f), math.ceil(f)
        return finite[lo] + (f - lo) * (finite[hi] - finite[lo])
    return {"n": len(values), "finite": len(finite),
            "mean": statistics.fmean(finite) if finite else None,
            "abs_mean": statistics.fmean(map(abs, finite)) if finite else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "q25": quantile(.25), "median": quantile(.5), "q75": quantile(.75)}


def relation(rows, y, *, log=False, within_uid=False):
    points = [(r["traj_uid"], float(r["L"]), float(r[y])) for r in rows
              if math.isfinite(r[y]) and (not log or r[y] > 0)]
    if log:
        points = [(uid, math.log(x), math.log(z)) for uid, x, z in points]
    if within_uid:
        grouped = {}
        for uid, x, z in points:
            grouped.setdefault(uid, []).append((x, z))
        centered = []
        for group in grouped.values():
            xm = statistics.fmean(x for x, _ in group)
            ym = statistics.fmean(z for _, z in group)
            centered.extend((x - xm, z - ym) for x, z in group)
    else:
        xm = statistics.fmean(x for _, x, _ in points)
        ym = statistics.fmean(z for _, _, z in points)
        centered = [(x - xm, z - ym) for _, x, z in points]
    xx = math.fsum(x*x for x, _ in centered)
    yy = math.fsum(z*z for _, z in centered)
    xy = math.fsum(x*z for x, z in centered)
    return {"n": len(points), "UIDs": len({u for u, _, _ in points}),
            "slope": xy/xx if xx else None,
            "pearson": xy/math.sqrt(xx*yy) if xx and yy else None,
            "x": "log(source-token length)" if log else "source-token length",
            "y": "log(" + y + ")" if log else y,
            "within_UID_centered": within_uid,
            "scope": "Descriptive association, not a causal length intervention"}


FIELDS = ("L", "d_sum", "d_abs_sum", "d_abs_mean", "A_sum", "A_abs_sum",
          "A_abs_mean", "d_cancellation", "A_cancellation", "native_root",
          "native_root_abs", "saved_root_abs")


def summarize(rows):
    return {"responses": len(rows), "UIDs": len({r["traj_uid"] for r in rows}),
            "source_tokens": sum(r["L"] for r in rows),
            "fields": {f: describe(r[f] for r in rows) for f in FIELDS},
            "token_pooled_mean_abs_d": math.fsum(r["d_abs_sum"] for r in rows) / sum(r["L"] for r in rows),
            "token_pooled_mean_abs_A": math.fsum(r["A_abs_sum"] for r in rows) / sum(r["L"] for r in rows)}


def length_account(rows):
    length = describe(r["L"] for r in rows)
    lower = [r for r in rows if r["L"] <= length["q25"]]
    upper = [r for r in rows if r["L"] >= length["q75"]]
    log = {f: relation(rows, f, log=True) for f in
           ("d_abs_sum", "d_abs_mean", "A_abs_sum", "A_abs_mean", "native_root_abs")}
    within = {f: relation(rows, f, log=True, within_uid=True) for f in
              ("d_abs_sum", "d_abs_mean", "A_abs_sum", "A_abs_mean")}
    return {
        "summary": summarize(rows),
        "length_vs": {f: relation(rows, f) for f in
                      ("d_abs_sum", "d_abs_mean", "A_abs_sum", "A_abs_mean", "d_cancellation", "native_root_abs")},
        "log_length_vs": log, "within_UID_log_length_vs": within,
        "exact_descriptive_identities": {
            "max_abs_mean_d_minus_L1_over_L": max(abs(r["d_abs_mean"]-r["d_abs_sum"]/r["L"]) for r in rows),
            "max_abs_mean_A_minus_L1_over_L": max(abs(r["A_abs_mean"]-r["A_abs_sum"]/r["L"]) for r in rows),
            "log_slope_d_mean_minus_L1_plus_1": log["d_abs_mean"]["slope"]-log["d_abs_sum"]["slope"]+1,
            "log_slope_A_mean_minus_L1_plus_1": log["A_abs_mean"]["slope"]-log["A_abs_sum"]["slope"]+1,
            "meaning": "Mean absolute coefficient is L1/L by definition. This is not an extra implementation division or a proposed normalization."},
        "lower_observed_quartile_including_ties": summarize(lower),
        "upper_observed_quartile_including_ties": summarize(upper),
    }


def length_contrast(account):
    lower = account["lower_observed_quartile_including_ties"]
    upper = account["upper_observed_quartile_including_ties"]
    return {"lower_responses": lower["responses"], "upper_responses": upper["responses"],
            "upper_over_lower_response_mean": {
                f: upper["fields"][f]["mean"]/lower["fields"][f]["mean"]
                for f in ("L", "d_abs_sum", "d_abs_mean", "A_abs_sum", "A_abs_mean", "native_root_abs")},
            "scope": "Observed length quartile contrast including ties, not matched length interventions"}


def main():
    mapped_path = ROOT / "readout-quality-20261006/v2/original-readout-mapped-requests.json"
    cases_path = ROOT / "readout-quality-20261006/v2/original-first-response-cases.json"
    batch_path = ROOT / "native-minibatch-v4/native-minibatch-batch-only-analysis.json"
    boundary_path = ROOT / "native-minibatch-v4/native-minibatch-update-boundary.json"
    gradient_path = ROOT / "native-minibatch-v4/task-signal-magnitude-source-audit.json"
    scale_path = ROOT / "native-minibatch-v4/textcraft-readout-signal-scale.json"
    curve_paths = [ROOT / f"author-reward-curve-20261006/v1/rank{r}-curves.json" for r in (0, 1)]
    requests = read(mapped_path)["requests"]
    rows, replicas = {}, {}
    for request in requests:
        key = (request["traj_uid"], request["env_step"])
        saved = request["saved_actor_slices"][0]
        if len(request["saved_actor_slices"]) != 1 or not saved["full_response_retained"]:
            raise ValueError("This analysis requires the recorded complete native response slice")
        L = request["source_end"]-request["source_start"]
        d, A = saved["d_from_saved_A"], saved["saved_A_summary"]
        if d["count"] != L or A["count"] != L or saved["actual_G"] != 1:
            raise ValueError("Original retained source length/reward does not match its credit summary")
        row = {"traj_uid": key[0], "source_step": key[1], "L": L,
               "d_sum": d["signed_sum"], "d_abs_sum": d["abs_sum"], "d_abs_mean": d["abs_mean"],
               "A_sum": A["signed_sum"], "A_abs_sum": A["abs_sum"], "A_abs_mean": A["abs_mean"],
               "d_cancellation": 1-abs(d["signed_sum"])/d["abs_sum"],
               "A_cancellation": 1-abs(A["signed_sum"])/A["abs_sum"],
               "native_root": request["trace"]["root_effect"],
               "native_root_abs": abs(request["trace"]["root_effect"]),
               "saved_root_abs": abs(d["signed_sum"]), "zero_d_tokens": d["zero_count"],
               "original_transport_rank": request["rank"], "original_request_index": request["request_index"]}
        replicas.setdefault(key, []).append(row)
        rows.setdefault(key, row)
    unique = list(rows.values())
    by_uid = {}
    for row in unique:
        by_uid.setdefault(row["traj_uid"], []).append(row)
    UID_summaries = [{"traj_uid": uid, "responses": len(rs), "source_tokens": sum(r["L"] for r in rs),
                      "d_sum": math.fsum(r["d_sum"] for r in rs),
                      "d_L1": math.fsum(r["d_abs_sum"] for r in rs),
                      "A_sum": math.fsum(r["A_sum"] for r in rs),
                      "A_L1": math.fsum(r["A_abs_sum"] for r in rs),
                      "token_mean_abs_d": math.fsum(r["d_abs_sum"] for r in rs)/sum(r["L"] for r in rs),
                      "token_mean_abs_A": math.fsum(r["A_abs_sum"] for r in rs)/sum(r["L"] for r in rs)}
                     for uid, rs in by_uid.items()]
    cases = [c for rank in read(cases_path)["rank_cases"] for c in rank]
    boundary = read(boundary_path)
    for c in cases:
        if boundary["traj_uid"][c["actor_row"]] != c["traj_uid"]:
            raise ValueError("Saved case does not match the original native actor row")
    first_G1 = [rows[(c["traj_uid"], c["source_step"])] for c in cases if c["actual_G"] == 1]
    first_A = {c["traj_uid"]: c for c in cases if c["actual_G"] == 1}
    raw_sums = []
    for row in first_G1:
        c = first_A[row["traj_uid"]]
        for field, saved_field in (("d_sum", "saved_source_d_from_A"), ("A_sum", "saved_source_A")):
            raw_sums.append(abs(row[field]-math.fsum(c[saved_field])))
    curves = {}
    for path in curve_paths:
        raw = read(path)
        if raw["phase"] != "complete_author_reward_curves":
            raise ValueError("Author metric raw file is not complete")
        for c in raw["cases"]:
            curves.setdefault(c["traj_uid"], c["curves"])
    first_curve_rows = []
    for row in first_G1:
        curve = curves[row["traj_uid"]]
        for view, v in curve["views"].items():
            points = v["score_points"]
            first_curve_rows.append({"traj_uid": row["traj_uid"], "L": row["L"], "view": view,
                "native_full_delete": points[-1]["native_factual_minus_deleted_logp"],
                "saved_full_delete": points[-1]["same_set_saved_d_sum"],
                "native_max_abs_cumulative_effect": max(abs(p["native_factual_minus_deleted_logp"]) for p in points),
                "saved_max_abs_cumulative_effect": max(abs(p["same_set_saved_d_sum"]) for p in points),
                "original_author_return": v["author_return"]})
    batch = read(batch_path)["native_minibatch"]
    source_account = read(gradient_path)
    scale = read(scale_path)
    all_account = length_account(unique)
    first_account = length_account(first_G1)
    positive = batch["by_saved_training_return"]["return_positive"]
    zero = batch["by_saved_training_return"]["return_zero"]
    result = {
        "scope": "Saved native64 response-credit length accounting. Descriptive, no model, normalization or new quality gate.",
        "created_unix": time.time(),
        "sources": [source(p) for p in [__file__, mapped_path, cases_path, batch_path, boundary_path, gradient_path, scale_path, *curve_paths,
                                        Path(__file__).resolve().parents[3]/"experiments/rl/PLAN.md",
                                        Path(__file__).resolve().parents[3]/"experiments/rl/RUNTIME_RECORD.md",
                                        Path(__file__).resolve().parents[3]/"experiments/rl/current_runtime.json"]],
        "coverage": {"original64_UIDs": len(cases), "G1_UIDs": len(by_uid), "G0_UIDs_without_d": sum(c["actual_G"]==0 for c in cases),
                     "readout_transport_requests": len(requests), "unique_G1_responses": len(unique),
                     "unique_response_source_tokens": sum(r["L"] for r in unique),
                     "duplicate_transports": len(requests)-len(unique),
                     "one_exact_zero_d_token_each_response": all(r["zero_d_tokens"] == 1 for r in unique)},
        "replica_policy": "First original mapped transport occurrence, no averaging; duplicates retained as ranges below.",
        "all_G1_responses": all_account, "first_G1_response_per_UID": first_account,
        "observed_quartile_contrasts": {"all186_G1_responses": length_contrast(all_account),
                                       "first21_G1_responses": length_contrast(first_account)},
        "first_response_lengths_by_realized_G": {str(g): {"UIDs": sum(c["actual_G"]==g for c in cases),
            "source_length": describe(c["source_end"]-c["source_start"] for c in cases if c["actual_G"]==g),
            "observed_length513_count": sum(c["actual_G"]==g and c["source_end"]-c["source_start"]==513 for c in cases)} for g in (0, 1)},
        "UID_balanced_success_account": {f: describe(u[f] for u in UID_summaries) for f in
            ("responses", "source_tokens", "d_sum", "d_L1", "A_sum", "A_L1", "token_mean_abs_d", "token_mean_abs_A")},
        "whole_native64_carrier_token_lengths": {
            "G1_rows": positive["row_count"], "G1_action_tokens": positive["action_token_count"],
            "G1_action_tokens_per_row_mean": positive["action_token_count"]/positive["row_count"],
            "G0_rows": zero["row_count"], "G0_action_tokens": zero["action_token_count"],
            "G0_action_tokens_per_row_mean": zero["action_token_count"]/zero["row_count"],
            "G1_fraction_of_original_carrier_action_tokens": positive["action_token_count"]/batch["all_action_tokens"]["action_token_count"],
            "scope": "Full retained policy-token trajectory exposure, not just first response, and not the equal-B4 PPO loss denominator"},
        "existing_all186_G1_token_distribution_verbatim": scale["token_weighted"],
        "formula_amplitude_description": {
            "pooled_A_L1_over_saved_d_L1": math.fsum(r["A_abs_sum"] for r in unique)/math.fsum(r["d_abs_sum"] for r in unique),
            "scope": "Same G=1 tokens, stored A versus inverse d; descriptive nonlinear-combination magnitude, not a new formula or acceptance test"},
        "native64_original_token_pooled_account": {
            "scope": "Original whole-carrier pooled counts, distinct from equal-B4 loss denominator; not a causal gradient factor.",
            "G1": batch["by_saved_training_return"]["return_positive"],
            "G0": batch["by_saved_training_return"]["return_zero"],
            "all": batch["all_action_tokens"]},
        "existing_native_gradient_and_B4_account_verbatim": {
            "coefficient_source_account": source_account["coefficient_source_account"],
            "gradient_account": source_account["gradient_account"]},
        "raw_first21_token_array_vs_mapping_max_sum_difference": max(raw_sums),
        "author_same_set_first21": first_curve_rows,
        "per_response_summaries": unique, "per_UID_summaries": UID_summaries,
        "replica_ranges": [{"traj_uid": uid, "step": step, "transports": len(rs),
            "ranges": {f: [min(r[f] for r in rs), max(r[f] for r in rs)] for f in FIELDS}}
            for (uid, step), rs in replicas.items() if len(rs)>1],
        "interpretation_limits": [
            "L1/L decomposition describes measured allocation; it is not evidence of a DT-only hidden division, equal credit broadcasting or a mandatory1/L law.",
            "A sublinear observed L1-length slope indicates sublinear credit mass growth in this population, not causal harm from length alone.",
            "G1-only response summaries do not describe failure d; G0 d is missing, whereas its stored sampled A is zero by the fixed formula.",
            "Signed sums can cancel positive/negative coefficients; their ratios are not gradient cancellation or causal error shares.",
            "Saved d is the FP64 inverse of stored FP32 A, not the pre-storage fresh DT vector.",
            "Native author curves evaluate model event scores for cumulative sets; they do not establish every token value or external-world accuracy.",
            "The GRPO comparison uses official group-standardized coefficients. The1/211 norm ratio is not an update ratio or an independent quality ratio.",
            "Within-UID associations do not hold source step, factual response content, query or actual length truncation fixed.",
            "No learned head, value model, rescaling, normalization, clipping, query change or method repair is applied here."],
        "operations": {"GPU": 0, "model": 0, "DT": 0, "gradient": 0, "optimizer": 0, "production_changes": 0}}
    out = ROOT / "native-minibatch-v4/response-credit-length-decomposition.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": source(out), "coverage": result["coverage"],
                      "all_log_slopes": result["all_G1_responses"]["log_length_vs"],
                      "within_UID_log_slopes": result["all_G1_responses"]["within_UID_log_length_vs"],
                      "first_lengths": result["first_response_lengths_by_realized_G"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
