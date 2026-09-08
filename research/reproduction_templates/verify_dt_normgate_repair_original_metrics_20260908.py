"""CPU-only audit of frozen NI1/MH1 paired norm-gate experiments.

Run with Python and NumPy. Read snapshots; write only the adjacent audit JSON.
Never import Torch, score a model, regenerate inputs, or load large GPU tensor
artifacts. Assertions deliberately fail on source, curve or ledger mismatches.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
R = A.parent / "DeltaTrace"
SNAPSHOT = A / "snapshot"
DEST = Path(__file__).with_suffix(".json")
spec = importlib.util.spec_from_file_location(
    "mas_diagnosis", R / "research/reproduction_templates/diagnose_NI0_MAS_raw_response.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
auc = mod.auc


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frozen(remote_path, expected):
    path = SNAPSHOT / remote_path.lstrip("/")
    raw = path.read_bytes()
    assert sha(raw) == expected, path
    return path, raw


def close(actual, expected, tolerance=1e-10):
    assert np.isfinite(actual) and np.isfinite(expected)
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def needle(w, keep, gold, recorded):
    gold = set(gold) & set(keep)
    if not gold:
        assert recorded is None
        return {"value": None, "reason": "No retained gold mapping; no needle score claimed."}
    k = max(1, math.ceil(len(keep) * .1))
    values = np.maximum(w[keep], 0)
    threshold = np.partition(values, -k)[-k]
    above = {keep[i] for i in np.flatnonzero(values > threshold)}
    ties = {keep[i] for i in np.flatnonzero(values == threshold)}
    selected_ties = k - len(above)
    base_hits, tie_gold = len(above & gold), len(ties & gold)
    min_hits = base_hits + max(0, selected_ties - (len(ties) - tie_gold))
    max_hits = base_hits + min(selected_ties, tie_gold)
    assert min_hits == max_hits, "Needle cutoff tie has ambiguous gold membership"
    close(min_hits / len(gold), recorded, 0)
    return {"value": recorded, "hits": min_hits, "gold_after_keep": len(gold),
            "top_k": k, "cutoff": float(threshold), "cutoff_ties": len(ties)}


def curve_audit(row, w, info, gold, needle_recorded):
    keep = info["keep"]
    assert w.dtype == np.float32 and w.shape == (info["prompt_length"],)
    assert np.isfinite(w).all() and len(keep) == len(set(keep))
    order = row["sorted_keep"]
    assert len(order) == len(keep) and set(order) == set(keep)
    assert np.all(np.diff(w[order]) <= 0), "Recorded order is not descending"
    attr_sum = float(w[keep].sum(dtype=np.float32))
    assert np.isclose(attr_sum, row["attr_sum"], rtol=1e-6, atol=0)
    offset, deleted, density = 0, set(), [1.]
    assert len(row["input_receipts"]) == 21
    for step, receipt in enumerate(row["input_receipts"]):
        if step:
            size = len(keep) // 20 + (step <= len(keep) % 20)
            group = order[offset:offset + size]
            offset += size
            deleted.update(group)
            density.append(density[-1] - float(w[group].sum(dtype=np.float32)) / row["attr_sum"]
                           if row["attr_sum"] > 0 else 1 - step / 20)
        assert receipt["deleted_positions"] == sorted(deleted)
    assert offset == len(keep)
    density_error = float(np.max(np.abs(np.asarray(density) - row["density"])))
    assert density_error < 2e-6
    for name in ("scores", "density", "normalized_model_response", "alignment_penalty", "corrected_scores"):
        assert len(row[name]) == 21 and np.isfinite(row[name]).all()
    scores = np.asarray(row["scores"], dtype=np.float64)
    assert scores[0] != scores[-1]
    response = np.minimum.accumulate(np.clip((scores - scores[-1]) / abs(scores[0] - scores[-1]), 0, 1))
    penalty = np.abs(response - np.asarray(row["density"], dtype=np.float64))
    corrected = np.clip(response + penalty, 0, 1)
    spread = corrected.max() - corrected.min()
    corrected = ((corrected - corrected.min()) / spread if spread else np.linspace(1, 0, 21))
    errors = {name: float(np.max(np.abs(value - row[name]))) for value, name in (
        (response, "normalized_model_response"), (penalty, "alignment_penalty"), (corrected, "corrected_scores"))}
    assert max(errors.values()) < 1e-12
    metrics = [auc(response), auc(corrected), auc(response + penalty)]
    metric_error = float(np.max(np.abs(np.asarray(metrics) - row["return_metrics"])))
    assert metric_error < 1e-12
    diagnosis = mod.diagnose(row)
    return {"original_RISE": metrics[0], "original_MAS": metrics[1],
            "original_unclipped_corrected_AUC": metrics[2], "alignment_AUC": auc(penalty),
            "needle": needle(w, keep, gold, needle_recorded),
            "FP32_attr_sum_numpy": attr_sum, "FP32_attr_sum_recorded": row["attr_sum"],
            "FP32_attr_sum_absolute_difference": abs(attr_sum - row["attr_sum"]),
            "density_reconstruction_max_absolute_difference": density_error,
            "formula_max_absolute_differences": errors, "metric_max_absolute_difference": metric_error,
            "deletion_steps_verified": 20, "deleted_tokens_verified": len(deleted),
            "native_endpoint_scores": [float(scores[0]), float(scores[-1])],
            "density_outside_unit_interval_AUC": diagnosis["density_outside_unit_interval_AUC"],
            "negative_predicted_group_observations": diagnosis["negative_predicted_group_observations"]}


def audit_job(tag):
    directory = SNAPSHOT / f"tmp/codex_dt_normgate_repair_{tag}_20260908_v1"
    r = json.loads((directory / "results.json").read_bytes())
    p = json.loads((directory / "protocol.json").read_bytes())
    receipt = json.loads((directory / "terminal_receipt.json").read_bytes())
    assert p == r["protocol"] and r["status"] == f"{tag}_paired_normgate_repair_original_metrics_complete"
    if "proc_exists" in receipt:
        assert receipt["proc_exists"] is False
    else:
        assert receipt["status"] == r["status"]
    checked, unavailable = {}, []
    for name, entry in receipt["files"].items():
        path = directory / name
        if not path.exists():
            unavailable.append(name)
            continue
        digest = sha_file(path)
        assert digest == entry["sha256"] and path.stat().st_size == entry["bytes"], name
        checked[name] = digest
    assert all(name in checked for name in ("results.json", "protocol.json", "vectors.npz", "review_bundle.zip"))
    with zipfile.ZipFile(directory / "review_bundle.zip") as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert Path(name).name == name
            assert (directory / name).read_bytes() == archive.read(name), name
    assert r["sources_before"] == r["sources_after"]
    assert r["weight_stats_before"] == r["weight_stats_after"] == p["expected_weight_stats"]
    for name, expected in p["files_sha256"].items():
        raw = (directory / name).read_bytes()
        assert sha(raw) == expected == r["sources_before"][name], name
        ast.parse(raw)
    for name, expected in p["unchanged_finite_runtime_sha256"].items():
        assert sha((directory / name).read_bytes()) == expected
    for name, expected in p["runtime_source_sha256"].items():
        assert r["sources_before"]["native/" + name] == expected
    official = R / "research/third_party/flashtrace_qwen35_e81b3be"
    for name, expected in p["official_source_blob_sha1"].items():
        raw = (official / name).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == expected
        assert sha(raw) == r["sources_before"]["FT/" + name]
    assert {k:r[k] for k in ("model_loads", "DT_calls", "FT_calls", "generation_calls", "scoring_forwards", "extra_conv_preactivation_calls")} == {
        "model_loads":1, "DT_calls":1, "FT_calls":0, "generation_calls":0, "scoring_forwards":46, "extra_conv_preactivation_calls":1}
    quality_path, quality_raw = read_frozen(p["quality_parent_path"], p["quality_parent_sha256"])
    quality = json.loads(quality_raw)
    quality_vectors = quality_path.with_name("vectors.npz")
    assert sha(quality_vectors.read_bytes()) == quality["vectors_sha256"]
    ftvec = np.load(quality_vectors, allow_pickle=False)
    _, prior_raw = read_frozen(p["parent_results_path"], p["parent_results_sha256"])
    prior = json.loads(prior_raw)
    parent_vector_path, _ = read_frozen(p["parent_vectors_path"], p["parent_vectors_sha256"])
    oldvec = np.load(parent_vector_path, allow_pickle=False)
    vec = np.load(directory / "vectors.npz", allow_pickle=False)
    assert len(r["cases"]) == 1
    key, case = next(iter(r["cases"].items()))
    info, old = case["input"], quality["cases"][key]
    assert case["gold"] == old["input"]["gold"]
    assert all(old["input"][name] == value for name, value in info.items())
    methods = {}
    for name, curve in case["curves"].items():
        w = vec[key + "_" + name + "_evaluated"]
        full = vec[key + "_" + name + "_full"]
        assert full.shape == (info["total_length"],) and np.isfinite(full).all()
        assert np.array_equal(w, full[:info["prompt_length"]].astype(np.float32))
        methods[name] = curve_audit(curve, w, info, case["gold"], curve["needle"])
    assert set(methods) == {"current", "candidate"}
    for name, curve in case["prior_fixed_FT_metrics"].items():
        assert curve == old["curves"][name]
        methods[name] = curve_audit(curve, ftvec[key + "_" + name], info, case["gold"], old["FT"]["needle"][int(name[-1])])
    assert set(case["prior_fixed_FT_metrics"]) == {"FT0", "FT1", "FT2", "FT3"}
    for curve in list(case["curves"].values()) + list(case["prior_fixed_FT_metrics"].values()):
        assert curve["input_receipts"][0]["input_sha256"] == info["input_sha256"]
        assert curve["input_receipts"][-1]["input_sha256"] == old["input"]["baseline_sha256"]
    fixed = {}
    assert sorted(map(int, case["scoring_points"])) == p["capture_steps"] == [0, 1, 10, 20]
    for label, point in case["scoring_points"].items():
        step = int(label)
        assert point["input_receipt"] == prior["cases"][key]["curve"]["input_receipts"][step]
        assert point["prior_original_native_score"] == prior["cases"][key]["curve"]["scores"][step]
        assert point["capture"]["decoder_calls"] == {k:1 for k in ("input_norm", "post_norm", "gate", "up", "silu", "down", "mlp", "decoder")}
        assert point["capture"]["mixer_calls"] == {"module":1, "conv":1, "FLA":1, "stage":1}
        assert point["capture"]["initial_cache"]["has_previous_state"] is False
        if not step:
            continue
        actual = case["scoring_points"]["0"]["original_native_score"] - point["original_native_score"]
        close(actual, point["complete_input"]["actual_model_effect"], 0)
        fixed[label] = {"actual_model_effect":actual, "deleted_count":len(point["input_receipt"]["deleted_positions"])}
        for name in ("current", "candidate"):
            deleted = point["input_receipt"]["deleted_positions"]
            pred = float(vec[key + "_" + name + "_evaluated"][deleted].astype(np.float64).sum())
            close(pred, point["complete_input"][name]["prediction"])
            close(pred - actual, point["complete_input"][name]["prediction_minus_actual"])
            ledger = point[name + "_layer0_decomposition"]
            residual = ledger["output_contraction"] - ledger["input_contraction"]
            close(residual, ledger["actual_minus_predicted"])
            close(sum(ledger["terms"].values()) - residual, ledger["telescoping_error"])
            assert abs(ledger["telescoping_error"]) < 1e-7
            fixed[label][name] = {"prediction":pred, "prediction_minus_actual":pred-actual, "absolute_error":abs(pred-actual),
                                 "layer0_actual_minus_predicted":residual, "layer0_terms":ledger["terms"]}
        assert point["current_layer0_decomposition"]["output_contraction"] == point["candidate_layer0_decomposition"]["output_contraction"]
        fixed[label]["candidate_minus_current_absolute_error"] = fixed[label]["candidate"]["absolute_error"] - fixed[label]["current"]["absolute_error"]
    dt, paired = case["DT_with_paired_diagnostics"], case["paired_repair"]
    close(float(vec[key + "_current_full"].sum()), dt["signed_sum"])
    assert len(dt["layers"]) == 32
    assert all(layer["replay_relative_L2"] == 0 and layer.get("FA_auxiliary_relative_L2",0) == 0 for layer in dt["layers"].values())
    kinds = [row["kind"] for row in dt["calls"]]
    assert all(sum(kind.startswith(prefix) for kind in kinds) == count for prefix,count in (("native_replay_",32),("finite_decoder_",32),("public_FA_LSE_",8)))
    calls = case["finite_FLA_calls"]
    assert len(calls) == 25 and [v["kind"] for v in calls] == ["current"] * 24 + ["candidate"]
    assert all(v["status"] == "returned" for v in calls)
    assert [i for i,v in enumerate(calls) if v["captured_layer0"]] == [23,24]
    assert calls[-1]["scale"] == calls[-2]["scale"] == paired["actual_scale"]
    assert calls[-1]["do_shape"] == calls[-2]["do_shape"]
    assert paired["status"] == "layer0_symmetric_normgate_computed"
    assert paired["mnorm_check"]["passed"] and paired["mnorm_check"]["bitwise_equal"]
    assert paired["mnorm_check"]["relative_L2"] == 0
    assert paired["fixed_coefficient_reuse"]["m_mixer_output_same_object"] and paired["fixed_coefficient_reuse"]["m_mlp_norm_output_same_object"]
    assert all(t["finite"] and t["device"] == "cuda:0" and ("expected_shape" not in t or t["shape"] == t["expected_shape"]) for t in paired["tensor_checks"].values())
    assert all(c["status"] == "returned" and c["end_synchronized"] and c["wall_seconds"] > 0 for c in paired["calls"])
    work = paired["additional_work"]
    ones = ("finite_FLA_calls_entered", "finite_FLA_calls_returned", "GDN_repropagations_entered", "GDN_repropagations_returned", "public_paired_conv_preactivation_calls", "public_conv_autograd_calls", "input_norm_residual_calls_entered", "input_norm_residual_calls_returned")
    assert all(work[name] == 1 for name in ones) and work["native_FLA_adjoint_stage_calls"] == 2
    assert all(work[name] == 0 for name in ("extra_model_forwards", "finite_FA_calls", "MLP_repropagations", "postnorm_repropagations", "other_layer_repropagations"))
    for name in ("current", "candidate"):
        ledger = case["B2_" + name]
        close(ledger["input_contraction"], float(vec[key + "_" + name + "_full"].sum()))
        close(ledger["input_contraction"], paired["B2_endpoint_contractions"]["decoder"][name]["total"])
        close(ledger["output_contraction"], paired["B2_endpoint_contractions"]["decoder"]["actual_effect"]["total"])
        close(sum(ledger["terms"].values()), ledger["actual_minus_predicted"])
    historical = float(np.linalg.norm(vec[key + "_current_full"] - oldvec[key + "_DT_full"]) / np.linalg.norm(oldvec[key + "_DT_full"]))
    close(historical, case["historical_vector_relative_L2_report_only"])
    report = {"status":"passed", "case":key, "input":info, "gold_count_raw":len(set(case["gold"])),
              "results_sha256":checked["results.json"], "vectors_sha256":checked["vectors.npz"],
              "methods":methods, "fixed_deletion_effects":fixed,
              "candidate_minus_current":{metric:methods["candidate"][metric]-methods["current"][metric] for metric in ("original_RISE","original_MAS")},
              "candidate_minus_frozen_FT":{name:{metric:methods["candidate"][metric]-methods[name][metric] for metric in ("original_RISE","original_MAS")} for name in ("FT0","FT1","FT2","FT3")},
              "historical_vector_relative_L2_report_only":historical,
              "paired_metadata":{name:paired[name] for name in ("mnorm_check","additional_work","calls","complete_diagnostic_seconds","cost_contract")},
              "metadata_validation_scope":"Saved GPU checks and scalar ledgers validated; large paired coefficient tensors are not reloaded or recomputed.",
              "native_source_scope":"Recorded before/after identities checked against protocol; native installed binary/source files are not executed by this CPU audit.",
              "locally_checked_receipt_files":checked, "receipt_artifacts_unavailable_locally":unavailable}
    return report, p, (directory / "study.py").read_text(encoding="utf-8")


def main():
    ni, nip, ni_study = audit_job("NI1")
    mh, mhp, mh_study = audit_job("MH1")
    assert set(nip["files_sha256"]) == set(mhp["files_sha256"])
    common = {name:value for name,value in nip["files_sha256"].items() if name != "study.py"}
    assert all(mhp["files_sha256"][name] == value for name,value in common.items())
    substitutions = (
        ("Paired layer0 norm-gate repair on one original NI1 example.", "Same frozen paired layer0 norm-gate repair on original MH1."),
        ("assert p['case_indices']==[['niah_mq_q2',1]]", "assert p['case_indices']==[['morehopqa',1]]"),
        ("NI1 development comparison only. Original new-order curves measured. No MH confirmation or production speed claim; no automatic promotion or multi-rule sweep.", "MH1 existing-case confirmation only, not an independent holdout. Original new-order curves measured. No independent confirmation or production speed claim; no automatic promotion or multi-rule sweep."),
        ("NI1_paired_normgate_repair_original_metrics_complete", "MH1_paired_normgate_repair_original_metrics_complete"),
    )
    for old,new in substitutions:
        assert ni_study.count(old) == 1
        ni_study = ni_study.replace(old,new)
    assert ni_study == mh_study, "Unreviewed NI/MH study computation difference"
    for name in ("runtime_source_sha256", "official_source_blob_sha1", "native_model_sha256", "installed_FA_interface_sha256", "checkpoint_config_tokenizer_sha256", "expected_weight_stats", "native_stage_source_sha256"):
        assert nip[name] == mhp[name], name
    report = {"status":"NI1_MH1_original_metrics_independently_audited", "audit_script_sha256":sha(Path(__file__).read_bytes()),
              "extra_model_calls":0, "sample_count":2, "independent_holdout":False,
              "jobs":{"NI1":ni,"MH1":mh}, "identical_computational_files_sha256":common,
              "study_difference_contract":"Only dataset selection and descriptive title/scope/status strings differ; exact remaining text verified.",
              "negative_evidence":[{"case":job["case"],"step":int(step),"increase_in_absolute_error":point["candidate_minus_current_absolute_error"]}
                                   for job in (ni,mh) for step,point in job["fixed_deletion_effects"].items() if point["candidate_minus_current_absolute_error"]>0],
              "limits":["NI1 and MH1 are historically used development examples, not independent holdouts; keep datasets separate.",
                        "Twenty deletion steps within one trajectory are correlated, not independent samples. BF16 score ties remain unresolved.",
                        "Own-order MAS/RISE curves and fixed-deletion conditional errors answer different questions; improved MAS does not imply every fixed conditional error improves.",
                        "FT0-3 comparisons use hash-verified frozen same-case original scores, not new FT reruns or aggregate benchmark evidence.",
                        "NumPy versus Torch FP32 reduction-order differences use the existing 1e-6 relative attr-sum and 2e-6 density tolerances; metric formulas use 1e-12 absolute tolerance.",
                        "Paired diagnostic duplication and nested timings are not a production speed comparison; no automatic candidate promotion follows."]}
    DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"negative_evidence":report["negative_evidence"],
                      "methods":{tag:{name:{k:v[k] for k in ("original_RISE","original_MAS","needle")} for name,v in job["methods"].items()} for tag,job in report["jobs"].items()}},ensure_ascii=False))


if __name__ == "__main__":
    main()
