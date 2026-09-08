"""Audit eight saved production DT calls without any model or GPU execution.

Uses the original-metric NumPy audit and actual saved per-run evaluated vectors.
Identical bin input hashes, not bitwise vector equality, determine score reuse.
Only the adjacent production summary JSON is written.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
D = A / "snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1"
DEST = A / "dt_normgate_production_summary_20260908.json"
spec = importlib.util.spec_from_file_location("paired_audit", A / "verify_dt_normgate_repair_original_metrics_20260908.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def vector_check(actual, reference, recorded):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert list(actual.shape) == recorded["shape"]
    assert recorded["dtype"] == {np.dtype("float32"):"torch.float32", np.dtype("float64"):"torch.float64"}[actual.dtype]
    exact = actual.tobytes() == reference.tobytes()
    assert exact == recorded["bitwise_equal"]
    assert audit.sha(actual.tobytes()) == recorded["actual_sha256"]
    assert audit.sha(reference.tobytes()) == recorded["reference_sha256"]
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    norm, denominator = float(np.linalg.norm(delta)), float(np.linalg.norm(reference.astype(np.float64)))
    audit.close(norm, recorded["difference_L2"])
    audit.close(float(np.max(np.abs(delta))), recorded["max_absolute"])
    assert int(np.sum(actual != reference)) == recorded["changed_elements"]
    if denominator:
        audit.close(norm / denominator, recorded["relative_L2"])
    else:
        assert recorded["relative_L2"] is None
    return {"bitwise_equal":exact, "relative_L2":recorded["relative_L2"],
            "max_absolute":recorded["max_absolute"], "changed_elements":recorded["changed_elements"]}


def audit_bins(w, info, bins):
    keep, order = info["keep"], bins["sorted_keep"]
    assert len(order) == len(keep) and set(order) == set(keep)
    assert np.all(np.diff(w[order]) <= 0)
    assert len(bins["groups"]) == 20 and len(bins["input_receipts"]) == 21
    offset, deleted = 0, set()
    assert bins["input_receipts"][0] == {"input_sha256":info["input_sha256"], "deleted_positions":[]}
    for step, group in enumerate(bins["groups"]):
        size = len(keep) // 20 + (step < len(keep) % 20)
        assert group == order[offset:offset + size]
        offset += size
        deleted.update(group)
        assert bins["input_receipts"][step+1]["deleted_positions"] == sorted(deleted)
    assert offset == len(keep)


def costs(r):
    out = {"measured_NI":{}, "initial_calls":r["initial_call_cost"],
           "compiler_context":r["compiler_context"], "scope":r["cost_scope"]}
    for method in ("current", "candidate"):
        rows = [row for row in r["runs"] if row["case"] == "niah_mq_q2_1" and row["phase"] == "measured" and row["method"] == method]
        assert len(rows) == 2
        source = r["NI_measured_cost"][method]
        assert source["samples"] == 2 and source["run_numbers"] == [row["number"] for row in rows]
        times = [row["production_details"]["complete_attribution_seconds_with_diagnostics"] for row in rows]
        peaks = [row["production_details"]["peak_allocated"] for row in rows]
        before = [row["GPU_allocated_before"] for row in rows]
        after = [row["GPU_allocated_after_cleanup"] for row in rows]
        assert source["complete_attribute_seconds"] == times and source["peak_allocated_bytes"] == peaks
        assert source["GPU_allocated_before_bytes"] == before
        audit.close(statistics.median(times), source["median_complete_attribute_seconds"], 0)
        audit.close(statistics.median(peaks), source["median_peak_allocated_bytes"], 0)
        out["measured_NI"][method] = {"samples":2, "run_numbers":source["run_numbers"], "seconds":times,
            "median_seconds":statistics.median(times), "peak_allocated_bytes":peaks, "median_peak_allocated_bytes":statistics.median(peaks),
            "allocated_before_bytes":before, "allocated_after_cleanup_bytes":after,
            "reserved_before_bytes":[row["GPU_reserved_before"] for row in rows],
            "incremental_peak_over_allocated_before_bytes":[peak-base for peak,base in zip(peaks,before)]}
    current, candidate = out["measured_NI"]["current"], out["measured_NI"]["candidate"]
    ratio = candidate["median_seconds"] / current["median_seconds"]
    audit.close(ratio, r["NI_measured_cost"]["candidate_over_current_latency_ratio"], 0)
    out["candidate_over_current_median_latency_ratio"] = ratio
    out["candidate_minus_current_median_peak_allocated_bytes"] = candidate["median_peak_allocated_bytes"] - current["median_peak_allocated_bytes"]
    out["equal_measured_allocated_baselines"] = len(set(current["allocated_before_bytes"] + candidate["allocated_before_bytes"])) == 1
    out["limits"] = "Two measured samples per method in C,S,S,C order are descriptive only. Initial rule/shape calls use existing compiler caches, not verified empty caches. No statistical speed guarantee or paired-diagnostic timing is used."
    initial = [row for row in r["runs"] if row["phase"] != "measured"]
    assert len(initial) == len(r["initial_call_cost"]) == 4
    for row, recorded in zip(initial,r["initial_call_cost"]):
        assert recorded == {"run_number":row["number"],"case":row["case"],"method":row["method"],
                            "complete_attribute_seconds":row["production_details"]["complete_attribution_seconds_with_diagnostics"],
                            "peak_allocated_bytes":row["production_details"]["peak_allocated"]}
    return out


def main():
    r = json.loads((D / "results.json").read_bytes())
    p = json.loads((D / "protocol.json").read_bytes())
    receipt = json.loads((D / "terminal_receipt.json").read_bytes())
    assert p == r["protocol"] == json.loads((A / "dt_normgate_production_protocol_20260908.json").read_bytes())
    if "proc_exists" in receipt:
        assert receipt["proc_exists"] is False
    checked = {}
    for name, entry in receipt["files"].items():
        path = D / name
        assert path.exists(), path
        assert path.stat().st_size == entry["bytes"] and audit.sha_file(path) == entry["sha256"], name
        checked[name] = entry["sha256"]
    assert all(name in checked for name in ("results.json","protocol.json","vectors.npz","review_bundle.zip"))
    with zipfile.ZipFile(D / "review_bundle.zip") as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert Path(name).name == name and (D / name).read_bytes() == archive.read(name)
    for name, expected in p["files_sha256"].items():
        raw = (D / name).read_bytes()
        assert audit.sha(raw) == expected
        ast.parse(raw)
        assert r["sources_before"][name] == expected
    # Guard the launch snapshot: one ordinary attribute call site, observer=None,
    # no import/call of either paired diagnostic helper or coefficient backend.
    tree = ast.parse((D / "study.py").read_bytes())
    callsites = [node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == "attribute"]
    assert len(callsites) == 1 and any(k.arg == "observer" and isinstance(k.value,ast.Constant) and k.value.value is None for k in callsites[0].keywords)
    assert "layer0_normgate_paired_repair_20260908.py" not in p["files_sha256"]
    assert "finite_fla_coefficient_capture_20260908.py" not in p["files_sha256"]
    references, old_vectors = {}, {}
    for key, ref in p["paired_references"].items():
        _, raw = audit.read_frozen(ref["results_path"],ref["results_sha256"])
        references[key] = json.loads(raw)
        path, _ = audit.read_frozen(ref["vectors_path"],ref["vectors_sha256"])
        old_vectors[key] = np.load(path,allow_pickle=False)
        for field in ("native_model_sha256","installed_FA_interface_sha256","runtime_source_sha256","native_stage_source_sha256","checkpoint_config_tokenizer_sha256","expected_weight_stats","official_source_blob_sha1"):
            assert references[key]["protocol"][field] == p[field], (key,field)
        for name, expected in p["files_sha256"].items():
            if name not in ("study.py","qwen35_dense_finite_runner.py"):
                assert references[key]["protocol"]["files_sha256"][name] == expected, (key,name)
    prior_budget = {"paired_experiments":len(references), "complete_DT_calls":sum(v["DT_calls"] for v in references.values()),
                    "native_scoring_forwards":sum(v["scoring_forwards"] for v in references.values()),
                    "native_eager_B1_diagnostics":sum(sum(row["kind"] == "original_order_native_eager_B1_diagnostic" for row in v["calls"]) for v in references.values()),
                    "extra_symmetric_GDN_propagations":sum(v["extra_conv_preactivation_calls"] for v in references.values()),
                    "scope":"Previously completed NI1/MH1 paired diagnostics; separate from this production cost job."}
    assert prior_budget["complete_DT_calls"] == 2 and prior_budget["native_scoring_forwards"] == 92
    budget = {"earlier_paired_runs":prior_budget,
              "new_production_job":{"DT_entered":r["DT_calls_entered"],"DT_returned":r["DT_calls"],
                                    "native_root_forwards":r["native_root_forwards"],"native_scoring_forwards":r["scoring_forwards"],
                                    "FT_calls":r["FT_calls"],"generation_calls":r["generation_calls"],
                                    "finite_callback_counts":r["finite_callback_counts"],"cached_original_score_reads":r["cached_original_score_reads"],
                                    "cached_metric_replays":r["cached_metric_replays"],"job_seconds":r["job_seconds"]}}
    assert r["scoring_forwards"] == r["FT_calls"] == r["generation_calls"] == 0
    assert r["DT_calls_entered"] <= 8 and r["DT_calls"] <= r["DT_calls_entered"]
    if r["status"] != "eight_call_production_integration_cost_complete":
        summary = {"status":"production_failure_receipt_audited","next":"Review the saved execution failure and partial paid work before proposing any separately bounded follow-up.",
                   "actual_budget":budget,"production_status":r["status"],"error":r.get("error"),"runs":r["runs"],"source_receipt_hashes":checked}
        DEST.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({k:summary[k] for k in ("status","next","actual_budget")}))
        return
    assert r["sources_before"] == r["sources_after"]
    assert r["weight_stats_before"] == r["weight_stats_after"] == p["expected_weight_stats"]
    assert r["native_model_source_after_sha256"] == p["native_model_sha256"]
    assert r["native_FA_interface_after_sha256"] == p["installed_FA_interface_sha256"]
    for name, expected in p["runtime_source_sha256"].items():
        assert r["sources_before"]["native/"+name] == expected
    official = audit.R / "research/third_party/flashtrace_qwen35_e81b3be"
    for name, expected in p["official_source_blob_sha1"].items():
        raw = (official/name).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == expected
        assert audit.sha(raw) == r["sources_before"]["FT/"+name]
    assert r["DT_calls_entered"] == r["DT_calls"] == r["native_root_forwards"] == len(r["runs"]) == 8
    assert r["finite_callback_counts"] == {"finite_FA":{"constructed":True,"entered":64,"returned":64}, "finite_FLA":{"constructed":True,"entered":192,"returned":192}}
    expected_counts = {"native_root":1,"native_decoder_replays":32,"finite_decoder_calls":32,"public_FA_auxiliary_calls":8,"finite_FA_calls":8,"finite_FLA_calls":24}
    assert r["actual_call_totals"] == {k:v*8 for k,v in expected_counts.items()}
    vec = np.load(D/"vectors.npz",allow_pickle=False)
    integration, quality, pending = [], {}, []
    score_reads, metric_replays = 0, 0
    for number, run in enumerate(r["runs"]):
        key, method, phase = run["case"], run["method"], run["phase"]
        assert [key,method,phase] == p["call_schedule"][number] and run["number"] == number
        assert run["status"] == "complete" and run["native_root_forwards"] == 1 and run["counts"] == expected_counts
        assert run["native_ledger_status"] == "completed_and_checked"
        assert run["finite_callback_counts"] == {"finite_FA":{"entered":8,"returned":8},"finite_FLA":{"entered":24,"returned":24}}
        details = run["production_details"]
        assert details["norm_gate_rules"] == p["production_norm_gate_rules"][method]
        kinds = [row["kind"] for row in details["calls"]]
        assert kinds.count("native_root_with_CPU_checkpoints") == 1
        assert all(sum(kind.startswith(prefix) for kind in kinds) == count for prefix,count in (("native_replay_",32),("finite_decoder_",32),("public_FA_LSE_",8)))
        assert len(details["layers"]) == 32
        for layer in details["layers"].values():
            assert layer["decoder_calls"] == {name:1 for name in ("input_norm","post_norm","gate","up","silu","down","mlp","decoder")}
            expected_mixer = ({"module":1,"interface":1,"native_varlen":0,"native_dense":1}
                              if layer["block_type"] == "full_attention" else {"module":1,"conv":1,"FLA":1,"stage":1})
            assert layer["mixer_calls"] == expected_mixer
            assert np.isfinite(layer["replay_relative_L2"]) and np.isfinite(layer.get("FA_auxiliary_relative_L2",0))
        assert run["outer_attribute_seconds"] >= details["complete_attribution_seconds_with_diagnostics"] > 0
        assert details["peak_allocated"] >= run["GPU_allocated_before"] > 0
        case, reference = r["cases"][key], references[key]["cases"][key]
        info = case["input"]
        assert info == reference["input"]
        prefix = key+"_"+method+"_run"+str(number)
        full, evaluated = vec[prefix+"_full"], vec[prefix+"_evaluated"]
        assert full.shape == (info["total_length"],) and evaluated.shape == (info["prompt_length"],)
        assert np.array_equal(full[:info["prompt_length"]].astype(np.float32),evaluated)
        audit.close(float(full.sum()),details["signed_sum"])
        drift = {name:vector_check(value,old_vectors[key][key+"_"+method+"_"+suffix],run["integration"][name])
                 for name,value,suffix in (("full_vector",full,"full"),("evaluated_vector",evaluated,"evaluated"))}
        prior_curve = reference["curves"][method]
        bins = run["integration"]["deletion_audit"]
        audit_bins(evaluated,info,bins)
        assert bins["input_receipts"][-1]["input_sha256"] == case["baseline_sha256"]
        same_bins = bins["input_receipts"] == prior_curve["input_receipts"]
        same_order = bins["sorted_keep"] == prior_curve["sorted_keep"]
        assert same_bins == run["integration"]["all_bin_input_receipts_equal"] == run["integration"]["cached_metric_reuse_eligible"]
        assert same_order == run["integration"]["sorted_keep_equal"]
        root = {name:vector_check(np.asarray(details[name],dtype=np.float64),np.asarray(reference["DT_with_paired_diagnostics"][name],dtype=np.float64),run["paired_root_output_comparison"][name]) for name in ("target_logp0","target_logp1")}
        effect_delta = details["root_effect"]-reference["DT_with_paired_diagnostics"]["root_effect"]
        audit.close(effect_delta,run["root_effect_difference_from_paired"],0)
        integration.append({"number":number,"case":key,"method":method,"phase":phase,**drift,
            "same_bin_inputs":same_bins,"same_sorted_order":same_order,"root_output_drift":root,"root_effect_difference":effect_delta,
            "max_native_replay_relative_L2":max(row["replay_relative_L2"] for row in details["layers"].values()),
            "max_FA_auxiliary_relative_L2":max(row.get("FA_auxiliary_relative_L2",0) for row in details["layers"].values())})
        cached = case["cached_metrics"][method]["production_runs"][str(number)]
        assert cached["all_original_bin_input_hashes_match"] == same_bins and cached["native_scoring_forwards"] == 0
        if not same_bins:
            assert cached["status"] == "bounded_original_metric_followup_needed" and "return_metrics" not in cached
            pending.append(number)
            continue
        assert cached["status"] == "original_function_replayed_with_new_density_and_hash_verified_cached_scores"
        assert cached["cached_original_score_reads"] == 21 and cached["scores"] == prior_curve["scores"]
        own_curve = dict(cached,input_receipts=bins["input_receipts"])
        own = audit.curve_audit(own_curve,evaluated,info,reference["gold"],cached["needle"])
        assert np.array_equal(np.asarray(cached["return_metrics"])-prior_curve["return_metrics"],cached["return_metrics_minus_paired"])
        assert np.array_equal(np.asarray(cached["density"])-prior_curve["density"],cached["density_minus_paired"])
        quality.setdefault(key,{}).setdefault(method,[]).append({"run_number":number,**own,"return_metrics_minus_paired":cached["return_metrics_minus_paired"],
            "score_provenance":"Hash-matched previously measured native scores; this production run recomputed its own metric density with the original function, without new scoring."})
        score_reads += 21
        metric_replays += 1
    assert score_reads == r["cached_original_score_reads"] <= 168
    assert metric_replays == r["cached_metric_replays"] <= 8
    assert len({run["expected_paired_input_sha256"] for run in r["runs"] if run["case"] == "niah_mq_q2_1"}) == 1
    assert len({run["expected_paired_input_sha256"] for run in r["runs"] if run["case"] == "morehopqa_1"}) == 1
    budget["new_production_job"]["actual_call_totals"] = r["actual_call_totals"]
    budget["new_production_job"]["native_eager_B1_diagnostics"] = sum(row["kind"] == "original_order_native_eager_B1_diagnostic" for row in r["calls"])
    assert budget["new_production_job"]["native_eager_B1_diagnostics"] == 1
    budget["all_three_GPU_jobs_so_far"] = {"complete_DT_calls":prior_budget["complete_DT_calls"]+r["DT_calls"],
        "native_scoring_forwards":prior_budget["native_scoring_forwards"]+r["scoring_forwards"],
        "native_eager_B1_diagnostics":prior_budget["native_eager_B1_diagnostics"]+1,
        "scope":"The two earlier paired jobs plus this production job only; cached score reads add no native scoring forwards. Any later scoring follow-up is separate."}
    prior_summary = json.loads((A/"verify_dt_normgate_repair_original_metrics_20260908.json").read_bytes())
    summary = {"status":"production_cost_and_cached_original_quality_audited" if not pending else "production_cost_audited_quality_followup_needed",
        "next":"Retain this bounded result and the opt-in candidate; no default promotion, universal repair, holdout, or speed guarantee follows." if not pending else "Changed deletion inputs require a separately bounded original-scoring follow-up for runs "+str(pending)+"; their production quality is not yet established.",
        "actual_budget":budget,"production_status":r["status"],"quality":quality,"integration_runs":integration,
        "pending_original_metric_followup_runs":pending,"costs":costs(r),
        "prior_paired_fixed_deletion_negative_evidence":prior_summary["negative_evidence"],
        "source_receipt_hashes":checked,"audit_script_sha256":audit.sha(Path(__file__).read_bytes()),
        "limitations":["Both NI1 and MH1 are historical development cases; eight production runs do not create eight independent examples.",
                       "Quality uses original cached native scores only after every deletion-bin input hash matches, with each production vector's own original metric density; bitwise vector equality is not required.",
                       "NI1/MH1 paired original scores were paid earlier (92 scoring forwards). This production job performs zero scoring forwards; cached scalar reads are not native scores computed again.",
                       "The original native default scoring precision and BF16 ties remain. Improved own-order MAS does not imply all fixed conditional errors improve; MH step1 worsened.",
                       "All source files and recorded native identities are audited; installed GPU binaries and model weights are not loaded by this local NumPy audit."]}
    DEST.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":summary["status"],"next":summary["next"],"actual_budget":budget,"costs":summary["costs"],"pending_runs":pending},ensure_ascii=False))


if __name__ == "__main__":
    main()
