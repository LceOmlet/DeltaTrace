"""Audit 42 actual MH scorer calls on the two saved production vectors.

NumPy and immutable evidence only: no Torch, model execution, attribution or
new scores. Historical FT differences are retained as unmatched raw references.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
D = A / "snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_MH_metrics_20260908_v1"
DEST = A / "dt_normgate_production_MH_metrics_summary_20260908.json"
spec = importlib.util.spec_from_file_location("paired_audit", A / "verify_dt_normgate_repair_original_metrics_20260908.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    r = json.loads((D/"results.json").read_bytes())
    p = json.loads((D/"protocol.json").read_bytes())
    receipt = json.loads((D/"terminal_receipt.json").read_bytes())
    assert r["protocol"] == p
    assert r["status"] == receipt["status"] == "production_MH_original_metrics_complete"
    assert receipt["proc_exists"] is False
    checked = {}
    for name, entry in receipt["files"].items():
        path = D/name
        assert path.stat().st_size == entry["bytes"] and audit.sha_file(path) == entry["sha256"], name
        checked[name] = entry["sha256"]
    with zipfile.ZipFile(D/"review_bundle.zip") as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert Path(name).name == name and (D/name).read_bytes() == archive.read(name)
    assert r["sources_before"] == r["sources_after"]
    assert r["weight_stats_before"] == r["weight_stats_after"] == p["expected_weight_stats"]
    for name, expected in p["files_sha256"].items():
        raw = (D/name).read_bytes()
        assert audit.sha(raw) == expected == r["sources_before"][name]
        ast.parse(raw)
    assert r["native_model_source_after_sha256"] == p["native_model_sha256"]
    assert r["native_FA_interface_after_sha256"] == p["installed_FA_interface_sha256"]
    for name, expected in p["runtime_source_sha256"].items():
        assert r["sources_before"]["native/"+name] == expected
    official = audit.R/"research/third_party/flashtrace_qwen35_e81b3be"
    for name, expected in p["official_source_blob_sha1"].items():
        raw = (official/name).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == expected
        assert audit.sha(raw) == r["sources_before"]["FT/"+name]
    _, raw = audit.read_frozen(p["production_results_path"],p["production_results_sha256"])
    production = json.loads(raw)
    vector_path, _ = audit.read_frozen(p["production_vectors_path"],p["production_vectors_sha256"])
    vectors = np.load(vector_path,allow_pickle=False)
    ref = p["paired_reference"]
    _, raw = audit.read_frozen(ref["results_path"],ref["results_sha256"])
    paired = json.loads(raw)
    for field in ("native_model_sha256","installed_FA_interface_sha256","runtime_source_sha256","native_stage_source_sha256","checkpoint_config_tokenizer_sha256","expected_weight_stats","official_source_blob_sha1"):
        assert p[field] == production["protocol"][field] == paired["protocol"][field]
    for name, expected in p["files_sha256"].items():
        if name != "study.py":
            assert production["protocol"]["files_sha256"][name] == expected
    expected_zero = ("DT_calls","FT_calls","finite_FA_calls","finite_FLA_calls","generation_calls")
    assert all(r[name] == 0 for name in expected_zero)
    assert r["model_loads"] == r["native_eager_diagnostics"] == 1
    assert r["scoring_forwards_entered"] == r["scoring_forwards_returned"] == 42
    assert p["case_indices"] == [["morehopqa",1]] and p["saved_production_run_numbers"] == {"current":6,"candidate":7}
    tree = ast.parse((D/"study.py").read_bytes())
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == "attribute" for node in ast.walk(tree))
    key = "morehopqa_1"
    assert set(r["cases"]) == {key}
    case = r["cases"][key]
    assert case["input"] == production["cases"][key]["input"] == paired["cases"][key]["input"]
    assert case["gold"] == paired["cases"][key]["gold"] == []
    assert case["prior_fixed_FT_metrics"] == paired["cases"][key]["prior_fixed_FT_metrics"]
    methods, endpoints, historical = {}, [], {}
    for method in ("current","candidate"):
        curve = case["curves"][method]
        number = p["saved_production_run_numbers"][method]
        run = production["runs"][number]
        assert curve["status"] == "original_actual_scores_complete"
        assert run["case"] == key and run["method"] == method and run["number"] == curve["production_run"] == number
        assert run["integration"]["cached_metric_reuse_eligible"] is False
        prefix = key+"_"+method+"_run"+str(number)
        w, full = vectors[prefix+"_evaluated"], vectors[prefix+"_full"]
        assert w.dtype == np.float32 and full.dtype == np.float64 and np.isfinite(full).all()
        assert np.array_equal(w,full[:case["input"]["prompt_length"]].astype(np.float32))
        assert audit.sha(w.tobytes()) == curve["source_vector_sha256"] == run["integration"]["evaluated_vector"]["actual_sha256"]
        bins = run["integration"]["deletion_audit"]
        assert curve["input_receipts"] == bins["input_receipts"]
        assert curve["sorted_keep"] == bins["sorted_keep"]
        assert curve["native_forwards_returned"] == 21 and curve["native_logits_dtype"] == "torch.bfloat16"
        assert curve["native_logits_shape"][:2] == [1,case["input"]["total_length"]]
        methods[method] = audit.curve_audit(curve,w,case["input"],case["gold"],curve["needle"])
        methods[method]["production_run"] = number
        methods[method]["source_vector_sha256"] = curve["source_vector_sha256"]
        old = paired["cases"][key]["curves"][method]
        actual_endpoints, old_endpoints = [curve["scores"][0],curve["scores"][-1]], [old["scores"][0],old["scores"][-1]]
        expected_endpoint_record = {"actual":actual_endpoints,"prior_paired":old_endpoints,
                                    "actual_minus_prior":[x-y for x,y in zip(actual_endpoints,old_endpoints)]}
        assert curve["endpoint_score_comparison"] == expected_endpoint_record
        assert case["prior_paired_metrics"][method] == old["return_metrics"]
        assert np.array_equal(np.asarray(curve["return_metrics"])-old["return_metrics"],curve["return_metrics_minus_paired"])
        endpoints.append((tuple(actual_endpoints),tuple(row["input_sha256"] for row in (curve["input_receipts"][0],curve["input_receipts"][-1]))))
        raw_differences = {}
        for name, frozen in case["prior_fixed_FT_metrics"].items():
            difference = (np.asarray(curve["return_metrics"])-frozen["return_metrics"]).tolist()
            assert difference == curve["return_metrics_minus_fixed_FT"][name]
            frozen_endpoints = [frozen["scores"][0],frozen["scores"][-1]]
            assert frozen["input_receipts"][0]["input_sha256"] == curve["input_receipts"][0]["input_sha256"]
            assert frozen["input_receipts"][-1]["input_sha256"] == curve["input_receipts"][-1]["input_sha256"]
            raw_differences[name] = {"return_metrics":frozen["return_metrics"],"native_endpoint_scores":frozen_endpoints,
                "new_minus_historical_metrics_raw_only":difference,
                "endpoint_scores_equal_to_new":frozen_endpoints == actual_endpoints,
                "matched_cross_run_win_claim_eligible":False}
        historical[method] = {"paired_endpoint_comparison":expected_endpoint_record,"frozen_FT_raw_references":raw_differences,
                             "reason":"Identical input hashes and sources do not erase changed native scoring endpoints. Historical FT values are not matched evidence for a current cross-run win."}
    assert len(set(endpoints)) == 1, "Current/candidate native scoring endpoints or their input hashes differ"
    assert endpoints[0][0] == (-81.5,-166.0)
    assert all(not ft["endpoint_scores_equal_to_new"] for row in historical.values() for ft in row["frozen_FT_raw_references"].values())
    assert all(row["paired_endpoint_comparison"]["prior_paired"] == [-82.0,-167.0] for row in historical.values())
    kinds = [row["kind"] for row in r["calls"]]
    assert kinds == ["actual_model_load","original_order_native_eager_B1_diagnostic","original_full_curve_current","original_full_curve_candidate"]
    assert all(row["seconds"] > 0 for row in r["calls"])
    production_summary = json.loads((A/"dt_normgate_production_summary_20260908.json").read_bytes())
    assert production_summary["source_receipt_hashes"]["results.json"] == p["production_results_sha256"]
    prior_budget = production_summary["actual_budget"]
    earlier = prior_budget["all_three_GPU_jobs_so_far"]
    actual_budget = {"preceding_paired_and_production_jobs":prior_budget,
        "this_MH_scoring_followup":{"model_loads":1,"native_eager_B1_diagnostics":1,"complete_DT_calls":0,
            "native_scoring_forwards_entered":42,"native_scoring_forwards_returned":42,"finite_FA_calls":0,"finite_FLA_calls":0,"FT_calls":0,"generation_calls":0,
            "job_seconds":r["job_seconds"]},
        "bounded_series_total":{"GPU_jobs":4,"complete_DT_calls":earlier["complete_DT_calls"],
            "native_scoring_forwards":earlier["native_scoring_forwards"]+42,"native_eager_B1_diagnostics":earlier["native_eager_B1_diagnostics"]+1,
            "scope":"Two paired jobs, eight-call production job, and this 42-score follow-up. Historical FT generation costs predate this series; cached reads are not native scoring."}}
    changes = {metric:methods["candidate"][metric]-methods["current"][metric] for metric in ("original_RISE","original_MAS")}
    summary = {"status":"production_MH_own_original_metrics_independently_audited",
        "next":"This bounded test series is complete. Keep the goal active and the candidate opt-in; any validation of historical MH decoder19/6 localization requires a new actual pass under a separate budget. No further scoring or FT rerun is required to interpret this within-evaluation result.",
        "actual_budget":actual_budget,"methods":methods,"candidate_minus_current":changes,
        "native_scoring_endpoint_control":{"same_within_current_evaluation":True,"both_methods":list(endpoints[0][0]),
            "historical_paired_and_FT":[-82.0,-167.0],"new_minus_historical":[.5,1.0],"historical_FT_matched_win_claim_eligible":False},
        "historical_metrics_raw_only":historical,
        "production_quality_followup_resolved_runs":[6,7],
        "production_cost_summary_reference":{"file":"dt_normgate_production_summary_20260908.json","sha256":audit.sha((A/"dt_normgate_production_summary_20260908.json").read_bytes()),
            "interpretation":"Its historical pending MH metric status is resolved by this later audit; cost samples and their limits are unchanged."},
        "source_receipt_hashes":checked,"audit_script_sha256":audit.sha(Path(__file__).read_bytes()),
        "limits":["Own current/candidate improvements are validated on the saved production vectors and native scores from this same evaluation, with matching endpoint controls. No historical FT winner claim is made.",
            "This is one historically used MH development example; no holdout or universal repair claim follows. NI and MH stay separate.",
            "Original BF16 scoring precision, score ties and twenty correlated deletion steps remain. The prior MH fixed-step1 worsening remains negative evidence, not invalidated by better own-order MAS.",
            "Production latency has only two measured NI samples per method; retain the prior descriptive cost result, not a speed guarantee.",
            "Native source hashes and recorded weight metadata are audited locally; no GPU model, scorer or finite attribution code is executed by this NumPy analyzer."]}
    DEST.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":summary["status"],"next":summary["next"],"methods":{name:{k:v[k] for k in ("original_RISE","original_MAS","needle")} for name,v in methods.items()},
                      "candidate_minus_current":changes,"endpoint_control":summary["native_scoring_endpoint_control"],"series_budget":actual_budget["bounded_series_total"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
