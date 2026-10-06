"""Describe the completed author cumulative reward curves and draw two figures.

No model, DT, gradients, new metric definition, or quality acceptance gate.
Original author metrics are read verbatim; saved d is not fresh DT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


VIEWS = ("signed_RISE", "positive_MAS")
DEFAULT_ROOT = (Path(__file__).resolve().parent / "textcraft-degradation-20261005"
                / "author-reward-curve-20261006" / "v1")


def artifact(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = fraction * (len(values) - 1)
    left = math.floor(index)
    right = math.ceil(index)
    return values[left] + (index - left) * (values[right] - values[left])


def stats(values):
    values = list(values)
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {"count": len(values), "finite_count": len(finite),
            "nonfinite_or_missing_count": len(values) - len(finite),
            "mean": statistics.fmean(finite) if finite else None,
            "abs_mean": statistics.fmean(map(abs, finite)) if finite else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "q25": quantile(finite, .25), "median": quantile(finite, .5),
            "q75": quantile(finite, .75)}


def paired_summary(points):
    pairs = [(float(p["same_set_saved_d_sum"]),
              float(p["native_factual_minus_deleted_logp"])) for p in points
             if all(math.isfinite(p[k]) for k in
                    ("same_set_saved_d_sum", "native_factual_minus_deleted_logp"))]
    estimated, native = [p[0] for p in pairs], [p[1] for p in pairs]
    differences = [y - x for x, y in pairs]
    if len(pairs) > 1:
        xm, ym = statistics.fmean(estimated), statistics.fmean(native)
        xx = math.fsum((x - xm) ** 2 for x in estimated)
        yy = math.fsum((y - ym) ** 2 for y in native)
        xy = math.fsum((x - xm) * (y - ym) for x, y in pairs)
        pearson = xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else None
    else:
        pearson = None
    opposite = [(x, y) for x, y in pairs if x * y < 0]
    native_mass = math.fsum(abs(y) for _, y in pairs)
    return {
        "count": len(points), "finite_pair_count": len(pairs),
        "saved_cumulative_d": stats(estimated), "native_cumulative_delta_logp": stats(native),
        "native_minus_saved": stats(differences),
        "mae": statistics.fmean(map(abs, differences)) if pairs else None,
        "rmse": math.sqrt(statistics.fmean(d * d for d in differences)) if pairs else None,
        "pearson": pearson,
        "sign_counts": {"same_nonzero": sum(x * y > 0 for x, y in pairs),
                        "opposite": len(opposite),
                        "saved_zero_native_nonzero": sum(x == 0 and y != 0 for x, y in pairs),
                        "native_zero_saved_nonzero": sum(y == 0 and x != 0 for x, y in pairs),
                        "both_zero": sum(x == 0 and y == 0 for x, y in pairs)},
        "opposite_sign_native_abs_mass_fraction": (
            math.fsum(abs(y) for _, y in opposite) / native_mass if native_mass else None),
        "scope": "Descriptive same cumulative deletion sets; nested points are not independent observations",
    }


def extract_record(rank, ordinal, case):
    curves = case["curves"]
    contract = curves["case_contract"]
    views = {}
    for view in VIEWS:
        original = curves["views"][view]
        points = []
        for point in original["score_points"]:
            points.append({
                key: point[key] for key in
                ("point_index", "raw_logp", "same_set_saved_d_sum",
                 "native_factual_minus_deleted_logp", "author_deleted_source_indices",
                 "author_current_group_source_indices", "author_deleted_input_positions")})
            points[-1]["deleted_source_count"] = len(point["author_deleted_source_indices"])
            points[-1]["deleted_source_fraction"] = (
                len(point["author_deleted_source_indices"]) / contract["source_tokens"])
        views[view] = {"author_return": original["author_return"],
                       "author_return_fields": original["author_return_fields"],
                       "author_curve": original["author_curve"], "score_points": points,
                       "native_score_calls": original["native_score_calls"],
                       "attribution_view": original["attribution_view"]}
    return {"rank": rank, "transport_ordinal": ordinal,
            "case_index": case["case_index"], "traj_uid": case["traj_uid"],
            "source_step": int(case["source_step"]),
            "observed_return": case["observed_return"], "case_contract": contract,
            "selected_metrics_verbatim": curves["metrics"], "views": views}


def replica_description(records):
    result = {"transport_records": [{k: r[k] for k in
              ("rank", "transport_ordinal", "case_index")} for r in records], "views": {}}
    for view in VIEWS:
        result["views"][view] = {
            "metric_ranges": [stats(r["views"][view]["author_return"][i] for r in records)
                              for i in range(3)],
            "per_point": [{
                "point_index": index,
                "raw_logp": stats(r["views"][view]["score_points"][index]["raw_logp"]
                                  for r in records),
                "native_delta_logp": stats(r["views"][view]["score_points"][index]
                                           ["native_factual_minus_deleted_logp"] for r in records),
                "saved_d_sum": stats(r["views"][view]["score_points"][index]
                                     ["same_set_saved_d_sum"] for r in records),
                "original_deleted_sets_equal": all(
                    r["views"][view]["score_points"][index]["author_deleted_source_indices"] ==
                    records[0]["views"][view]["score_points"][index]["author_deleted_source_indices"]
                    for r in records),
            } for index in range(len(records[0]["views"][view]["score_points"]))],
        }
    return result


def build_analysis(input_root):
    source_paths = [input_root / f"rank{rank}-curves.json" for rank in (0, 1)]
    ranks = [json.loads(p.read_text(encoding="utf-8")) for p in source_paths]
    if any(r["phase"] != "complete_author_reward_curves" for r in ranks):
        raise RuntimeError("Author raw results are not complete; no partial results are analyzed")
    records = [extract_record(rank, i, case) for rank, r in enumerate(ranks)
               for i, case in enumerate(r["cases"])]
    identities = {}
    for record in records:
        identities.setdefault((record["traj_uid"], record["source_step"]), []).append(record)
    primary = [replicas[0] for replicas in identities.values()]
    duplicates = [{"traj_uid": key[0], "source_step": key[1],
                   **replica_description(replicas)} for key, replicas in identities.items()
                  if len(replicas) > 1]
    summaries = {}
    for view in VIEWS:
        points = [p for record in primary for p in record["views"][view]["score_points"]
                  if p["point_index"] > 0]
        endpoints = [record["views"][view]["score_points"][-1] for record in primary]
        steps = sorted({p["point_index"] for record in primary
                        for p in record["views"][view]["score_points"]})
        step_summaries = []
        for step in steps:
            same_step = [(r, p) for r in primary for p in r["views"][view]["score_points"]
                         if p["point_index"] == step]
            rows = {"point_index": step, "unique_UID_count": len(same_step)}
            for field in ("raw_logp", "same_set_saved_d_sum", "native_factual_minus_deleted_logp",
                          "deleted_source_fraction"):
                rows[field] = stats(p[field] for _, p in same_step)
            for field in ("scores", "density", "normalized_model_response",
                          "alignment_penalty", "corrected_scores"):
                rows["author_" + field] = stats(r["views"][view]["author_curve"][field][step]
                                                for r, _ in same_step)
            step_summaries.append(rows)
        summaries[view] = {
            "metrics_verbatim_UID_summary": {field: stats(r["views"][view]["author_return"][i]
                                                          for r in primary)
                                             for i, field in enumerate(("rise", "mas", "rise_plus_ap"))},
            "cumulative_noninitial_points": paired_summary(points),
            "fully_deleted_endpoint": paired_summary(endpoints), "by_step": step_summaries,
        }
    return {
        "scope": "Completed original author metrics and same-set cumulative event LP effects; no quality gate",
        "created_unix": time.time(), "sources": {"analyzer": artifact(__file__),
            "raw_ranks": [artifact(p) for p in source_paths],
            "completed": artifact(input_root / "completed.json")},
        "original_runtime_sources": [r.get("sources") for r in ranks],
        "coverage": {"transport_cases": len(records), "unique_UID_source_steps": len(primary),
                     "unique_UIDs": len({r["traj_uid"] for r in primary}),
                     "duplicate_transport_cases": len(records) - len(primary),
                     "actual_native_forward_calls": [r["native_forward_calls"] for r in ranks],
                     "observed_return_values": sorted({r["observed_return"] for r in records})},
        "replica_policy": "Primary is first occurrence in rank0 then rank1, preserving each raw case order; official pad copy excluded from UID denominator; no replica averaging",
        "primary_records": primary, "replica_records": duplicates,
        "all_transport_records": records, "view_summaries": summaries,
        "limits": [
            "G1 saved-d cases only; no G0 d is fabricated, and this is not all186 responses or the full64 task population",
            "Saved d was inverted from stored FP32 A; it is not a fresh or pre-storage raw DT vector",
            "The target is the model's categorical reward-event LP, not an external-world counterfactual oracle",
            "Cumulative set effects are not independent single-token deletion values",
            "RISE/MAS are the original endpoint-normalized metrics and do not calibrate absolute token advantage or gradient scale",
            "Nested 20 deletion points and cases sharing original task prompts are not independent statistical samples",
            "All statistics are descriptive; no new tolerance, correction, training change, model call or gradient occurs",
        ],
    }


def draw_figures(analysis, output_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = analysis["primary_records"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    for column, view in enumerate(VIEWS):
        rows = analysis["view_summaries"][view]["by_step"]
        x = [r["point_index"] for r in rows]
        ax = axes[0, column]
        for field, label, color in (
            ("native_factual_minus_deleted_logp", "Native factual - deleted LP", "#176ca4"),
            ("same_set_saved_d_sum", "Saved DT sum over same set", "#d87921")):
            ax.plot(x, [r[field]["mean"] for r in rows], label=label, color=color, linewidth=1.7)
            ax.fill_between(x, [r[field]["q25"] for r in rows],
                            [r[field]["q75"] for r in rows], color=color, alpha=.14)
        ax.axhline(0, color=".65", linewidth=.7)
        ax.set(title=view.replace("_", " ") + " deletion order", ylabel="Cumulative effect (nats)")
        ax.legend(fontsize=8)
        ax = axes[1, column]
        for field, label, color in (
            ("author_normalized_model_response", "Author normalized response", "#176ca4"),
            ("author_density", "Author remaining attribution share", "#747474"),
            ("author_corrected_scores", "Author MAS corrected curve", "#d87921")):
            ax.plot(x, [r[field]["mean"] for r in rows], color=color, label=label, linewidth=1.5)
        ax.set(ylabel="Original author curve", xlabel="Deletion step (20 author groups)")
        ax.legend(fontsize=8)
        for ax in axes[:, column]:
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(alpha=.2)
    fig.suptitle(f"{len(primary)} unique successful UID; means, raw-effect shading = UID IQR\n"
                 "Official pad replica excluded; saved d is not fresh DT", fontsize=11)
    curve = output_root / "cumulative-deletion.png"
    fig.savefig(curve, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    for ax, view in zip(axes, VIEWS):
        points = [p for r in primary for p in r["views"][view]["score_points"]
                  if p["point_index"] > 0 and
                  math.isfinite(p["same_set_saved_d_sum"]) and
                  math.isfinite(p["native_factual_minus_deleted_logp"])]
        x = [p["same_set_saved_d_sum"] for p in points]
        y = [p["native_factual_minus_deleted_logp"] for p in points]
        scatter = ax.scatter(x, y, c=[p["deleted_source_fraction"] for p in points],
                             cmap="viridis", vmin=0, vmax=1, s=15, alpha=.65, edgecolors="none")
        values = x + y
        if values:
            bounds = [min(values), max(values)]
            ax.plot(bounds, bounds, color=".4", linestyle="--", linewidth=.9, label="Equal effects")
        ax.axhline(0, color=".7", linewidth=.7)
        ax.axvline(0, color=".7", linewidth=.7)
        ax.set(title=view.replace("_", " "), xlabel="Saved DT sum over deleted set (nats)",
               ylabel="Native factual - deleted LP (nats)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
        fig.colorbar(scatter, ax=ax, label="Actual source fraction deleted", shrink=.75)
    fig.suptitle(f"Same cumulative sets: {len(primary)} UID × 20 nested points per view\n"
                 "Descriptive event-score comparison; not independent token accuracy", fontsize=11)
    relation = output_root / "same-set-credit.png"
    fig.savefig(relation, dpi=180)
    plt.close(fig)
    return [artifact(curve), artifact(relation)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root).resolve()
    analysis = build_analysis(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    analysis["figures"] = draw_figures(analysis, output_root)
    path = output_root / "author-reward-curve-analysis.json"
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analysis": artifact(path), "coverage": analysis["coverage"],
                      "figures": analysis["figures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
