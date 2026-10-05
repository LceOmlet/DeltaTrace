"""Describe saved native readout cases by their original prompt UID groups.

Stdlib only: read the two completed rank JSONs and join their trajectory UIDs
to the saved original global64 update-boundary mapping. No model, environment
parser, trajectory reconstruction, credit generation or numerical gate.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time


HERE = Path(__file__).resolve().parent
DEGRADATION = HERE.parents[1]


def source(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def describe(values):
    values = [float(value) for value in values]
    assert values and all(math.isfinite(value) for value in values)
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                min=min(values), max=max(values), std_population=statistics.pstdev(values),
                abs_mean=statistics.fmean(abs(value) for value in values),
                rms=math.sqrt(statistics.fmean(value * value for value in values)),
                positive=sum(value > 0 for value in values),
                negative=sum(value < 0 for value in values), zero=sum(value == 0 for value in values))


def pearson(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)
            if xx and yy else None)


def float32(value):
    # The saved native target differences were torch.float32 subtractions.
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pool(rows):
    return dict(n=len(rows), success_count=sum(row["observed_return"] == 1 for row in rows),
                realized_success_fraction=statistics.fmean(row["observed_return"] for row in rows),
                factual_p_success=describe(row["factual_p_success"] for row in rows),
                full_response_eos_p_success=describe(row["full_response_eos_p_success"] for row in rows),
                p_success_factual_minus_full_eos=describe(row["p_success_factual_minus_full_eos"] for row in rows),
                full_span_observed_target_d=describe(row["full_span_observed_target_d"] for row in rows),
                factual_p_observed_class=describe(row["factual_p_observed_class"] for row in rows),
                full_eos_more_optimistic_count=sum(row["p_success_factual_minus_full_eos"] < 0 for row in rows),
                factual_argmax_success_count=sum(row["factual_argmax_class"] == 1 for row in rows),
                full_eos_argmax_success_count=sum(row["full_eos_argmax_class"] == 1 for row in rows))


def main():
    raw_paths = [HERE / f"rank{rank}-readout.json" for rank in (0, 1)]
    boundary_path = DEGRADATION / "native-minibatch-v4/native-minibatch-update-boundary.json"
    audit_path = DEGRADATION / "quality_interface_owner_audit.json"
    mapping_path = DEGRADATION / "native-minibatch-v4/native-minibatch-readout-mapping.json"
    prepared_path = HERE / "prepared.json"
    completed_path = HERE / "completed.json"
    inspection_path = HERE / "native-owner-inspection.json"
    boundary = json.loads(boundary_path.read_bytes())
    audit = json.loads(audit_path.read_bytes())
    mapping = json.loads(mapping_path.read_bytes())
    prepared = json.loads(prepared_path.read_bytes())
    completed = json.loads(completed_path.read_bytes())
    inspection = json.loads(inspection_path.read_bytes())
    assert boundary["rows"] == 64 and boundary["groups"] == 8
    assert len(boundary["traj_uid"]) == len(set(boundary["traj_uid"])) == 64
    uid_map = dict(zip(boundary["traj_uid"], boundary["uid"]))
    assert len(uid_map) == 64 and sorted(Counter(uid_map.values()).values()) == [8] * 8
    assert prepared["input"]["sha256"] == inspection["native_reader_inputs_sha256"]
    assert mapping["case_pack"]["artifact"]["sha256"] == prepared["input"]["sha256"]
    for name in ("producer", "readout"):
        expected = audit["sources"][name]["sha256"]
        basename = "deltatrace_rollout.py" if name == "producer" else "reward_readout.py"
        actual = [sha for path, sha in prepared["sources"].items() if path.endswith("/" + basename)]
        assert actual == [expected]
    rows, normalization_errors, native_sources = [], [], []
    for rank, path in enumerate(raw_paths):
        run = json.loads(path.read_bytes())
        assert run["rank"] == rank and run["phase"] == "complete_native_readout"
        assert run["optimizer_steps"] == run["scheduler_steps"] == run["backward_calls"] == run["finite_trace_calls"] == 0
        assert run["outcome_token_ids"] == [15, 16] and len(run["cases"]) == 32
        assert run["input_sha256"] == prepared["input"]["sha256"]
        assert [case["traj_uid"] for case in run["cases"]] == boundary["traj_uid"][rank * 32:(rank + 1) * 32]
        assert completed["ranks"][rank]["sha256"] == source(path)["sha256"]
        assert run["sources"]["producer"]["sha256"] == audit["sources"]["producer"]["sha256"]
        assert run["sources"]["runner"]["sha256"] == audit["sources"]["runner"]["sha256"]
        native_sources.append(dict(rank=rank, checkpoint=run["checkpoint"], sources=run["sources"],
                                   model_dtype=run["model_dtype"], native_fla_fp16=run["native_fla_fp16"]))
        for case in run["cases"]:
            assert case["variant_names"] == ["factual", "full_response_eos", "single_eos_0", "single_eos_1"]
            assert case["observed_return"] in (0, 1) and case["observed_class_index"] == int(case["observed_return"])
            assert case["source_step"] == 0 and case["native_logits_dtype"] == "torch.float32"
            logits = case["native_outcome_log_probs"]  # Actual saved two-class log probabilities.
            assert len(logits) == 4 and all(len(value) == 2 for value in logits)
            observed = case["observed_class_index"]
            target = [value[observed] for value in logits]
            assert target == case["native_target_log_probs"]
            assert float32(target[0] - target[1]) == case["native_full_span_log_ratio"]
            assert [float32(target[0] - target[i]) for i in (2, 3)] == case["native_single_log_ratios"]
            probabilities = [[math.exp(value) for value in variant] for variant in logits]
            normalization_errors.extend(abs(sum(variant) - 1) for variant in probabilities)
            length = case["source_end"] - case["source_start"]
            assert len(set(case["probe_source_positions"])) == 2
            assert all(0 <= position < length for position in case["probe_source_positions"])
            rows.append(dict(rank=rank, case_index=case["case_index"], prompt_uid=uid_map[case["traj_uid"]],
                             traj_uid=case["traj_uid"], source_step=case["source_step"],
                             observed_return=case["observed_return"], observed_class_index=observed,
                             source_start=case["source_start"], source_end=case["source_end"],
                             context_tokens=case["context_tokens"],
                             factual_p_success=probabilities[0][1], full_response_eos_p_success=probabilities[1][1],
                             p_success_factual_minus_full_eos=probabilities[0][1] - probabilities[1][1],
                             factual_p_observed_class=probabilities[0][observed],
                             factual_argmax_class=max(range(2), key=lambda i: logits[0][i]),
                             full_eos_argmax_class=max(range(2), key=lambda i: logits[1][i]),
                             full_span_observed_target_d=case["native_full_span_log_ratio"],
                             native_single_observed_target_ds=case["native_single_log_ratios"],
                             probe_source_positions=case["probe_source_positions"], probe_token_ids=case["probe_token_ids"]))
    assert len(rows) == len({row["traj_uid"] for row in rows}) == 64
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["prompt_uid"]].append(row)
    groups = []
    for uid, values in sorted(by_group.items()):
        split = {str(outcome): pool([row for row in values if row["observed_return"] == outcome])
                 for outcome in (0, 1) if any(row["observed_return"] == outcome for row in values)}
        groups.append(dict(prompt_uid=uid, **pool(values), by_observed_return=split,
                           success_minus_failure_factual_p_mean=(
                               split["1"]["factual_p_success"]["mean"] - split["0"]["factual_p_success"]["mean"]
                               if set(split) == {"0", "1"} else None)))
    successes = [row for row in rows if row["observed_return"] == 1]
    probes = [dict(rank=row["rank"], case_index=row["case_index"], prompt_uid=row["prompt_uid"],
                   traj_uid=row["traj_uid"], source_position=row["probe_source_positions"][index],
                   token_id=row["probe_token_ids"][index], native_single_d=value)
              for row in successes for index, value in enumerate(row["native_single_observed_target_ds"])]
    assert len(successes) == 21 and len(probes) == 42
    result = dict(
        scope=__doc__, created_unix=time.time(), status="descriptive_saved_readout_group_analysis_not_quality_repair",
        sources=[source(path) for path in [Path(__file__), *raw_paths, boundary_path, audit_path,
                                          mapping_path, prepared_path, completed_path, inspection_path]],
        actual_imports=native_sources,
        definitions=dict(group="Original native prompt UID, joined by unique trajectory UID; not a newly parsed environment task identifier.",
                         probability="math.exp of saved native FP32 two-class log probability; no renormalization or calibration.",
                         probability_delta="P(success|factual first response) minus P(success|full-response EOS).",
                         target_d="Saved native factual target log probability minus the deletion variant target log probability. Target class is success on return1 and failure on return0; the all pool mixes target classes.",
                         single_probe_scope="Standalone native read_outcomes B4 variants from readout-quality v2. Distinct from paired8 matched-layout/conditional-boundary probes; these values must not be substituted for that numerical path.",
                         statistics="Descriptive scalar reductions. Std is population std; signs refer to the stated scalar, not world-level action quality or a pass/fail threshold."),
        coverage=dict(rows=64, prompt_groups=8, trajectories=64, source_steps=dict(Counter(row["source_step"] for row in rows)),
                      observed_returns=dict(Counter(row["observed_return"] for row in rows)),
                      native_outcome_token_ids=[15, 16], target_selection_exact=True,
                      original_uid_order_exact=True, recorded_FP32_target_differences_exact=True,
                      max_probability_sum_minus_one_abs=max(normalization_errors)),
        all=pool(rows), by_observed_return={str(outcome): pool([row for row in rows if row["observed_return"] == outcome]) for outcome in (0, 1)},
        prompt_groups=groups,
        group_description=dict(all_group_means_exceed_this_batch_success_fraction=all(
            row["factual_p_success"]["mean"] > row["realized_success_fraction"] for row in groups),
            successful_mean_below_failed_mean_group_count=sum(row["success_minus_failure_factual_p_mean"] < 0 for row in groups if row["success_minus_failure_factual_p_mean"] is not None),
            group_mean_forecast_vs_group_success_pearson=pearson([row["factual_p_success"]["mean"] for row in groups], [row["realized_success_fraction"] for row in groups]),
            pooled_trajectory_forecast_vs_outcome_pearson=pearson([row["factual_p_success"] for row in rows], [row["observed_return"] for row in rows])),
        successful_first_response_single_probes=dict(scope="21 successful first responses, two previously selected fixed-random positions each; dependent pairs.",
            summary=describe(row["native_single_d"] for row in probes),
            by_prompt_uid={uid: describe(row["native_single_d"] for row in probes if row["prompt_uid"] == uid) for uid in sorted(by_group)}, probes=probes),
        source_contract=dict(producer=audit["sources"]["producer"], readout=audit["sources"]["readout"], runner=audit["sources"]["runner"],
            no_future_leakage_scope="Static audited source contract: query contains fixed rule/category legend, current step/horizon and sampling; observed return selects a target after its causal predictor. Diagnostic uses selected_input_ids[:-1]. Full64 input-ID pack is represented locally only by its original receipt SHA; this script does not claim per-ID dynamic leakage verification.",
            label_bias_scope="Only the fixed 0/1 label mapping was observed. Input-dependent forecasts rule out a constant response, but cannot isolate pure label-token preference from semantic/readout error.",
            actor_scope="Actual imported producer/native runner use the existing actor FSDP/PEFT object at checkpoint25, without an alternate value model or copied weights. No claim about every historical live adapter state is made."),
        rows=rows,
        limits=["Eight prompt groups and 64 first-response forecasts are one checkpoint-local sample, not an independent population calibration experiment.",
                "Whole-response EOS is a different intervention from a single-token deletion; observed-target signs across return0/1 are not one common success direction.",
                "High Brier, optimistic forecasts and weak within-group discrimination do not establish a reproducible interface or kernel bug.",
                "Native model probabilities are not an environment/world oracle; exact-token-value idealization remains unchanged.",
                "No environment parser, reconstructed rollout, credit replacement, new value head, model call, temperature change, threshold or production edit was used."])
    output = HERE / "independent-readout-groups.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=source(output), cases=len(rows), prompt_groups=len(groups),
                          full_eos_more_optimistic=result["all"]["full_eos_more_optimistic_count"],
                          success_probe_summary=result["successful_first_response_single_probes"]["summary"])))


if __name__ == "__main__":
    main()
