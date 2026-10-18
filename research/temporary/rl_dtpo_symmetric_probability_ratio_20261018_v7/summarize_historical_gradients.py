"""Summarize already collected native logs; no model or training operation.

This is analysis of archived receipts, not an alternative logger, loss,
normalizer, or method specification. PLAN.md remains the method specification.
"""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE / "receipts"
ROOT = HERE.parents[2]
OUT = Path("F:/DeltaTraceBackups/plots/dtpo-historical-gradient-audit-20261018-v7")
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(Path(path).read_bytes())

def provenance(path):
    path = Path(path)
    return {"path": str(path).replace("\\", "/"),
            "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}

native_path = RECEIPTS / "historical-version-native-log-metrics-v1.json"
audit_path = RECEIPTS / "historical-version-owner-source-audit-v1.json"
native, audit = read(native_path), read(audit_path)
current_v7_path = RECEIPTS / "textcraft-dt-current-source-snapshot.json"
current_v7 = read(current_v7_path)["jobs"][0]
assert current_v7["run_dir"] == native["jobs"][-1]["run_dir"]
native["jobs"][-1]["metrics"] = current_v7["metrics"]
labels = [
    ("v1 original", 2, "masked_whiten(r*u)", "all parsed Action targets"),
    ("pre-v3 group/no-whiten (actual run suffix v1)", 2, "g*u", "all parsed Action targets"),
    ("v3", 2, "g*u/std(g*u)", "all parsed Action targets"),
    ("v4 withdrawn", 2, "withdrawn linear weighted credit", "reward-settlement targets"),
    ("v5", 2, "g*u/std(g*u)", "reward-settlement targets"),
    ("v6 two GPU", 2, "g*u/std(u)", "reward-settlement targets"),
    ("v6 eight GPU (execution 20261018-v5)", 8, "g*u/std(u)", "reward-settlement targets"),
    ("v7 symmetric", 8, "g*tanh(d/2)", "reward-settlement targets"),
]
assert len(native["jobs"]) == len(labels)
norm_path = RECEIPTS / "native-norm-once-v7-v1/completed-accounting.json"
norm = read(norm_path)
raw_rank0 = norm["native_local_norm_mean_by_rank"][0]
rows, summaries = [], []

def stats(metrics, low=None, high=None):
    values = [m["values"]["actor/grad_norm"] for m in metrics
              if "actor/grad_norm" in m["values"]
              and (low is None or m["step"] >= low)
              and (high is None or m["step"] <= high)]
    return {"count": len(values), "median_console": statistics.median(values) if values else None,
            "min_console": min(values) if values else None,
            "max_console": max(values) if values else None}

def add(job, label, gpus, formula, target_scope):
    metrics = job.get("metrics", [])
    summary = {"label": label, "identity": job.get("identity", job.get("version")),
               "run_dir": job["run_dir"], "gpus": gpus, "credit_formula_summary": formula,
               "target_scope": target_scope, "completed_iterations": metrics[-1]["step"] if metrics else 0,
               "full_history": stats(metrics), "first_ten": stats(metrics, 1, 10),
               "first_three_console": [m["values"].get("actor/grad_norm") for m in metrics[:3]],
               "last_ten": stats(metrics, metrics[-1]["step"] - 9) if metrics else stats([]),
               "rounded_zero_steps": [m["step"] for m in metrics if m["values"].get("actor/grad_norm") == 0]}
    if label == "v7 symmetric":
        summary["raw_iteration3_first_returned_rank"] = raw_rank0
        summary["raw_iteration3_global_from_all_eight_original_norms"] = norm["materialized_global_norm_mean"]
        summary["rounded_zeros_are_not_exact_zeros"] = True
    summaries.append(summary)
    for m in metrics:
        refs = m.get("sources", [])
        rows.append({
            "label": label, "run_dir": job["run_dir"], "gpus": gpus, "iteration": m["step"],
            "native_console_grad_norm": m["values"].get("actor/grad_norm"),
            "native_raw_grad_norm": raw_rank0 if label == "v7 symmetric" and m["step"] == 3 else None,
            "native_raw_receipt": str(norm_path).replace("\\", "/") if label == "v7 symmetric" and m["step"] == 3 else None,
            "console_zero_is_rounded": m["values"].get("actor/grad_norm") == 0,
            "training_reward": m["values"].get("episode/reward/mean"),
            "entropy": m["values"].get("actor/entropy_loss"),
            "policy_loss": m["values"].get("actor/pg_loss"),
            "normalized_advantage_min": m["values"].get("critic/advantages/min"),
            "normalized_advantage_max": m["values"].get("critic/advantages/max"),
            "sources": json.dumps(refs, ensure_ascii=False),
        })

for job, (label, gpus, formula, target) in zip(native["jobs"], labels):
    add(job, label, gpus, formula, target)

controls_path = Path("F:/DeltaTraceBackups/plots/textcraft-dt-reward-target-exponential-variance-20261017-v6/20261017-183652/source-snapshot.json")
controls = read(controls_path)
for job in controls["jobs"]:
    if job["method"] in ("ppo", "grpo"):
        add(job, job["method"].upper() + " historical control", 2, "native fixed VERL " + job["method"], "native")
control_receipt = RECEIPTS / "historical-PPO-GRPO-native-metrics-v1.json"
control_receipt.write_text(json.dumps({
    "source": provenance(controls_path),
    "collected_at": controls["collected_at"],
    "jobs": [{k: j.get(k) for k in ("method", "version", "run_dir", "metrics")}
             for j in controls["jobs"] if j["method"] in ("ppo", "grpo")],
    "scope": "Existing collector snapshot; preserves the original PPO native resume-chain intervals.",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

v3_path = ROOT / "experiments/rl/results_textcraft_DTPO_variance_v3_20261015.json"
v3 = read(v3_path)["verification"]
group_path = ROOT / "research/temporary/rl_textcraft_gradient_rootcause_20261013_v1/native-DT-group-gradient-summary-v26.json"
group = read(group_path)
outlier_path = ROOT / "research/temporary/rl_textcraft_efficiency_20261015_v1/step24-saved-credit-CPU-audit.json"
outlier = read(outlier_path)
scale_path = RECEIPTS / "native-readout-advantage-scale-first-three-v1.json"
scale = read(scale_path)

actor = [j["sources"].get("verl/workers/actor/dp_actor.py", {}).get("SHA256")
         for j in audit["jobs"]]
core = [j["sources"].get("verl/trainer/ppo/core_algos.py", {}).get("SHA256")
        for j in audit["jobs"]]
assert len(set(actor)) == 1 and len(set(core)) == 1
old = native["jobs"][0]["metrics"]
windows = []
for low, high in ((1, 10), (31, 40), (61, 70), (76, 85)):
    selected = [m for m in old if low <= m["step"] <= high]
    windows.append({"iterations": [low, high], **stats(selected),
                    "reward_median": statistics.median(m["values"]["episode/reward/mean"] for m in selected)})

report = {
    "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "scope": "Historical TextCraft native gradient magnitudes v1 through symmetric v7; no new model/update tests.",
    "norm_scope": "Original first-returned rank's pre-clipping shard norm, averaged over original optimizer updates. Different GPU counts are not global-norm matched.",
    "version_identity_note": "The group-multiplier/no-whitening intermediate run is actually named 20261014-v1. No separate verified formal method-v2 run was found; do not invent one.",
    "zero_scope": "Console values use three decimal places. v3 iteration24 and v7 zero-formatted values are censored measurements, not exact zero.",
    "summaries": summaries,
    "v1_growth_windows": windows,
    "native_owner_audit": {
        "source": provenance(audit_path), "actor_SHA256": actor[0], "core_SHA256": core[0],
        "scope": "Files selected by each recorded launch; historical runtime binding evidence is preserved in the individual deployment receipts.",
        "conclusion": "No method-version change to the original actor/PPO core or loss reduction was found."
    },
    "existing_direct_evidence": {
        "old_v1_common_positive_group_component": {
            "source": provenance(group_path), "norms": group["parameter_gradient_statistics"]["norms"],
            "partition_residual_relative": group["gradient_partition_residual_over_full"],
            "scope": "Previously saved fixed late minibatch, not a fresh experiment or a complete causal reconstruction of the whole training trajectory.",
            "interpretation": "All-same-positive-reward groups retain the original DT component; official GRPO assigns zero advantage to these groups."
        },
        "pre_v3_to_v3_scale": {
            "source": provenance(v3_path),
            "saved_actual_batch_raw_std": v3["real_saved_batch_CPU"]["raw_masked_std"],
            "saved_actual_batch_native_scale": v3["real_saved_batch_CPU"]["scale"],
            "v3_first_actual_PPO_input_raw_std": v3["actual_native_first_PPO_input"]["raw_DT_masked_std"],
            "v3_first_actual_PPO_input_scale": v3["actual_native_first_PPO_input"]["variance_scale"],
            "v3_first_actual_PPO_input_output_std": v3["actual_native_first_PPO_input"]["actual_advantage_masked_std"],
            "interpretation": "Observed approximately fivefold advantage rescaling is consistent with native early norms .004-.005 versus .021-.028; not a claim of exact whole-gradient proportionality."
        },
        "v3_step24": {
            "source": provenance(outlier_path),
            "raw_minimum": outlier["steps"]["24"]["raw_minimum"],
            "largest_single_A_squared_fraction_of_traced_total": outlier["steps"]["24"]["largest_single_A_squared_fraction_of_traced_total"],
            "native_normalized_extremum": outlier["steps"]["24"]["native_logged_normalized_extremum"],
            "conditional_std_inference": outlier["steps"]["24"]["inferred_std_if_this_extremum_is_native_retained"],
            "scope": outlier["steps"]["24"]["omitted_rows_scope"],
            "interpretation": "A single extreme exponential term dominates traced squared advantage and inflates the batch variance scale, suppressing ordinary coefficients. This is distinct from v7's small signal from the outset."
        },
        "v7_actual_signal_scale": {"source": provenance(scale_path), "scope": scale["scope"], "summary": scale["summary"]},
        "v7_raw_native_norm": {"source": provenance(norm_path), "raw_rank0_iteration3": raw_rank0,
                              "global_iteration3": norm["materialized_global_norm_mean"],
                              "all_original_methods_restored": norm["original_methods_restored_all_eight_ranks"]},
    },
    "comparison": {
        "eight_GPU_v6_step3_native_console": native["jobs"][6]["metrics"][2]["values"]["actor/grad_norm"],
        "eight_GPU_v7_step3_native_raw_rank0": raw_rank0,
        "observed_ratio_v6_to_v7": native["jobs"][6]["metrics"][2]["values"]["actor/grad_norm"] / raw_rank0,
        "same_d_source_credit_identity": "A7/A6 = s_u/(1+exp(-d)), where A6=g*(1-exp(-d))/s_u. Applies to nonzero same-input source terms only; self-target is separate.",
        "small_d_limit": "A6 approximately g*d/s_u; A7 approximately g*d/2. Removing variance scaling and the bounded transform both alter magnitude.",
        "v6_vs_v5": "Normalizing u does not normalize g*u: Var(g*u)/Var(u) is not generally 1, because g/u dependence and zero-g rows remain.",
        "limits": [
            "Current v7 readout-scale summaries include full causal history and original RPC padding, not exactly the retained native PPO population.",
            "Advantage RMS ratios are not parameter-gradient norm ratios; token score directions, correlations and cancellation remain relevant.",
            "Different historical trajectories and normalization populations prevent assigning an exact percentage of the observed 18.46x gap to each change.",
            "No fresh v7 gradient-component decomposition was run; total norm also contains original entropy/KL terms.",
            "No exact retained-batch v6 standard-deviation receipt was located in the examined native logs; no value is invented.",
            "These magnitude observations do not establish policy quality or authorize rescaling to match a chosen norm range."
        ],
    },
    "provenance": [provenance(p) for p in (native_path, current_v7_path, audit_path, norm_path, scale_path, control_receipt)],
    "added_model_calls": 0, "added_backward_calls": 0, "added_optimizer_updates": 0,
    "training_method_or_hyperparameter_changes": 0, "current_six_run_goal": "active",
}
target = RECEIPTS / "historical-gradient-magnitude-analysis-v1.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / target.name).write_bytes(target.read_bytes())
csv_path = OUT / "historical-grad-norm-with-sources.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"analysis": str(target), "CSV": str(csv_path),
                  "summaries": [{k: s[k] for k in ("label", "completed_iterations", "full_history", "first_ten")} for s in summaries]},
                 ensure_ascii=False))
