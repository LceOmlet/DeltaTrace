"""Bind completed attribution diagnostics; no model calls or runtime changes."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/rl"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path):
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main():
    terminal_path = HERE / "native-head-terminal-environment.json"
    terminal = read(terminal_path)
    reports = {}
    for key in ("native_reference_drift", "native_reference_repeat",
                "native_head_textcraft", "native_head_appworld", "native_dt_lifecycle"):
        path = RESULTS / f"results_{key}_20261008.json"
        reports[key] = {"receipt": binding(path), "value": read(path)}

    tasks = {}
    for task in ("textcraft", "appworld"):
        value = reports[f"native_head_{task}"]["value"]
        tasks[task] = {
            "uniform": value["cohorts"]["uniform"]["summary"],
            "predicted_tail_census": value["cohorts"]["predicted_tail_census"]["summary"],
            "cohort_warning": "Remaining counts refer only to the previously missed-tail cohort; they are not a new population prevalence estimate.",
            "state_groups": value["cohorts"]["uniform"]["by_state"],
        }

    points = reports["native_head_textcraft"]["value"]["cohorts"]["uniform"]["points_with_identity"]
    missed = []
    for point in points:
        if point["old_DT_d"] >= 0 and point["old_native_d"] < -math.log(2):
            missed.append({
                "traj_uid": point["traj_uid"], "packed_slot": point["packed_slot"],
                "token_id": point["token_id"], "state_sha256": point["initial_state_sha256"],
                "original_DT_d": point["old_DT_d"], "native_d": point["native_d"],
                "FP32_head_d": point["FP32_d"],
                "isolated_head_rounding_d": point["isolated_output_rounding_d"],
                "FP32_head_advantage_per_reward": -math.expm1(-point["FP32_d"]),
            })
    assert len(missed) == 2

    jobs = []
    for name in ("native-head-textcraft-final", "native-head-appworld-final",
                 "native-dt-lifecycle-textcraft-final"):
        folder = HERE / name
        launch = read(folder / "launch.json")
        ranks = [read(folder / f"rank{rank}.json") for rank in (0, 1)]
        jobs.append({
            "name": name, "launch": binding(folder / "launch.json"),
            "pid": launch["pid"], "birth": launch["birth"],
            "source_sha256": launch["source_sha256"], "devices": launch["devices"],
            "base_commit_at_launch": launch["base_commit"],
            "files_not_committed_at_launch_bound_by_sha": True,
            "completion": binding(folder / "completed.json"),
            "effective_configuration": binding(folder / "effective-config.yaml"),
            "initialization": binding(folder / "actor-initialization.json"),
            "ranks": [{"receipt": binding(folder / f"rank{rank}.json"),
                       "owner_import_paths_and_hashes": record["owners"],
                       "worker_script_sha256": record["script_sha256"],
                       "operations": record["operations"],
                       "elapsed_seconds": record["elapsed_seconds"],
                       "peak_allocated_bytes": record.get("peak_allocated")}
                      for rank, record in enumerate(ranks)],
            "status": "completed_diagnostic_only_not_formal_deployment",
        })

    value = {
        "scope": "Extreme attribution evidence update, not learning-degradation diagnosis or a method replacement.",
        "upstream_VERL_commit": "20bd331",
        "collection": "The unchanged frozen 128 uniform sources and 37 predicted-tail sources per task; 330 paired points in total. Preserve task/state/cohort separation.",
        "reports": {k: v["receipt"] for k, v in reports.items()},
        "tasks": tasks, "persistent_TextCraft_missed_tail": missed,
        "lifecycle": {k: reports["native_dt_lifecycle"]["value"][k]
                      for k in ("trajectories", "states", "frozen_B4_groups", "query_rule", "phases")},
        "jobs": jobs,
        "terminal_environment": binding(terminal_path),
        "observed_unix": terminal["observed_unix"],
        "diagnostic_semantics": [
            "FP32 projection uses the same native hidden rows and head weight, not a full FP32 network or a new production readout.",
            "The native BF16 score remains the actual native-model score. Precision sensitivity does not establish real-world causal truth.",
            "No official FA/FLA/VERL numerical acceptance assertion is added or claimed by these diagnostics.",
            "The 45-trajectory lifecycle test excludes reproduced DT/observer forward-state pollution only in its measured scope; it does not certify gradients.",
            "All 330 inspected source contractions match their selected embedding boundary. This does not certify the entire actor interface.",
        ],
        "unresolved": [
            "Finite propagation still disagrees with robust native single-deletion effects; no collection-level root cause or accepted repair established.",
            "Earlier TextCraft layer-reference drift is not explained by the lifecycle test; do not attribute it to state pollution without evidence.",
            "Original actor nonfinite gradient and AppWorld formal OOM remain separate unresolved runtime issues.",
        ],
        "production_modified": False, "credit_correction_added": False,
        "formal_training_restarted": False, "checkpoint_restore": 0,
        "recorder": binding(Path(__file__)),
    }
    result_path = RESULTS / "results_extreme_attribution_evidence_20261008.json"
    write(result_path, value)

    runtime_path = RESULTS / "current_runtime.json"
    runtime = read(runtime_path)
    runtime["latest_native_reference_evidence_20261008"] = {
        "receipt": binding(result_path), "observed_unix": terminal["observed_unix"],
        "completed_diagnostic_jobs": [job["name"] for job in jobs],
        "production_modified": False, "fixed": False,
    }
    runtime["latest_runtime_status_20261008"].update({
        "observed_unix": terminal["observed_unix"],
        "source_receipt": binding(terminal_path), "research_processes_remaining": 0,
        "native_reference_diagnostics": "All three bounded probes complete; not deployed; no formal restart.",
        "physical_resources": "At the bound terminal observation all 8 GPUs 858/65536 MiB, 0%, no GPU processes.",
    })
    runtime["latest_observation_unix"] = terminal["observed_unix"]
    runtime["latest_observation_utc"] = datetime.fromtimestamp(
        terminal["observed_unix"], timezone.utc).isoformat()
    # Preserve old timestamped observations rather than rewriting them as current.
    write(runtime_path, runtime)
    print(json.dumps({"result": binding(result_path), "jobs": len(jobs),
                      "TextCraft_persistent_missed_tail": len(missed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
