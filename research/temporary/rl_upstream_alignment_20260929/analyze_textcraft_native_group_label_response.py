"""Read-only stdlib summary of saved native64 probabilities and label encoding.

No model/DT calls, loss changes, calibration, or acceptance thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time


AUDIT = Path(__file__).resolve().parent
DEG = AUDIT / "textcraft-degradation-20261005"
NATIVE = DEG / "readout-quality-20261006/v2"
LABEL = DEG / "label-encoding-20261006/v2"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def ids_hash(ids):
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


def describe(values):
    return {"count": len(values), "mean": sum(values) / len(values),
            "abs_mean": sum(abs(x) for x in values) / len(values),
            "min": min(values), "max": max(values)}


def separation(rows, key):
    pos = [r[key] for r in rows if r["G"] == 1]
    neg = [r[key] for r in rows if r["G"] == 0]
    n, p, g = len(rows), sum(r[key] for r in rows) / len(rows), sum(r["G"] for r in rows) / len(rows)
    pairs = len(pos) * len(neg)
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return {"rows": n, "success_rows": len(pos), "failure_rows": len(neg),
            "success_rate": g, "mean_probability": p,
            "mean_probability_success": sum(pos) / len(pos) if pos else None,
            "mean_probability_failure": sum(neg) / len(neg) if neg else None,
            "success_minus_failure": sum(pos) / len(pos) - sum(neg) / len(neg) if pairs else None,
            "Brier": sum((r[key] - r["G"]) ** 2 for r in rows) / n,
            "population_covariance_probability_reward": sum((r[key] - p) * (r["G"] - g) for r in rows) / n,
            "success_failure_pairs": pairs, "pairwise_wins_with_half_ties": wins,
            "AUC": wins / pairs if pairs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=NATIVE / "native64-task-group-label-response.json")
    args = parser.parse_args()
    paths = {"carrier_rows": NATIVE / "native-b4-denominator-analysis.json",
             "original_cases": NATIVE / "original-first-response-cases.json",
             "decoded_prefixes": NATIVE / "native-prefix-decoding.json",
             "official_handlers": NATIVE / "original-native-collect-metadata.json",
             "label_inspection": LABEL / "native-owner-inspection.json",
             **{f"native_rank{r}": NATIVE / f"rank{r}-readout.json" for r in (0, 1)},
             **{f"label_rank{r}": LABEL / f"rank{r}-readout.json" for r in (0, 1)}}
    data = {name: load(path) for name, path in paths.items()}
    cases = [c for rank in data["original_cases"]["rank_cases"] for c in rank]
    carrier = {r["traj_uid"]: r for r in data["carrier_rows"]["rows"]}
    decoded = {r["traj_uid"]: r for r in data["decoded_prefixes"]["cases"]}
    native = {c["traj_uid"]: c for r in (0, 1) for c in data[f"native_rank{r}"]["cases"]}
    label = {c["traj_uid"]: c for r in (0, 1) for c in data[f"label_rank{r}"]["cases"]}
    handler_index = {}
    for i, h in enumerate(data["official_handlers"]["handlers"]):
        handler_index.setdefault(ids_hash(h["turns"][0]["native_response_ids"]), []).append((i, h))
    assert len(cases) == len(carrier) == len(native) == len(label) == 64
    assert data["label_inspection"]["only_two_label_ids_changed"]
    rows, grouped = [], {}
    for c in cases:
        uid, saved = c["traj_uid"], carrier[c["traj_uid"]]
        nc, lc, dc = native[uid], label[uid], decoded[uid]
        ids, start, end = c["selected_input_ids"], c["source_start"], c["source_end"]
        G = c["actual_G"]
        assert G in (0, 1) and G == saved["saved_row_return"] == nc["observed_return"] == lc["observed_return"]
        assert c["source_step"] == nc["source_step"] == lc["source_step"] == 0
        assert start == nc["source_start"] == lc["source_start"] and end == nc["source_end"] == lc["source_end"]
        assert hashlib.sha256(json.dumps(ids[:start], separators=(",", ":")).encode()).hexdigest() == dc["prompt_ids_sha256"]
        matches = handler_index[ids_hash(ids[start:end])]
        assert len(matches) == 1
        hi, h = matches[0]
        assert h["score"] == G and not h["missing_fields"]
        a, b = lc["original_query_ids"], lc["swapped_query_ids"]
        changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert len(a) == len(b) and sorted((a[i], b[i]) for i in changed) == [(15, 16), (16, 15)]
        assert lc["success_class_indices"] == [1, 1, 0, 0]
        assert lc["observed_class_indices"] == ([1, 1, 0, 0] if G == 1 else [0, 0, 1, 1])
        lp, nlp = lc["native_outcome_log_probs"], nc["native_outcome_log_probs"]
        success_lp = [lp[i][k] for i, k in enumerate(lc["success_class_indices"])]
        observed_lp = [lp[i][k] for i, k in enumerate(lc["observed_class_indices"])]
        row = {"actor_row": c["actor_row"], "traj_uid": uid, "prompt_uid": saved["uid"],
               "official_handler_index_exact_response_ID_match": hi, "official_item_id": h["item_id"],
               "prompt_ids_int64_le_sha256": ids_hash(ids[:start]), "source_ids_int64_le_sha256": ids_hash(ids[start:end]),
               "source_step": 0, "source_tokens": end - start, "G": G,
               "native_v2_factual": math.exp(nlp[0][1]), "native_v2_fullEOS": math.exp(nlp[1][1]),
               **{key: math.exp(x) for key, x in zip(("original_factual", "original_fullEOS", "swapped_factual", "swapped_fullEOS"), success_lp)},
               "original_success_root_d": success_lp[0] - success_lp[1],
               "swapped_success_root_d": success_lp[2] - success_lp[3],
               "original_observed_root_d": observed_lp[0] - observed_lp[1],
               "swapped_observed_root_d": observed_lp[2] - observed_lp[3]}
        rows.append(row)
        grouped.setdefault(saved["uid"], []).append(row)
    keys = ("native_v2_factual", "native_v2_fullEOS", "original_factual", "original_fullEOS", "swapped_factual", "swapped_fullEOS")
    groups = []
    for uid, rs in grouped.items():
        assert len(rs) == 8 and len({r["prompt_ids_int64_le_sha256"] for r in rs}) == len({r["official_item_id"] for r in rs}) == 1
        prompt = decoded[rs[0]["traj_uid"]]["prompt_decode"]
        groups.append({"prompt_uid": uid, "official_item_id": rs[0]["official_item_id"],
                       "goal": re.search(r"Goal: craft (.*?)\.", prompt).group(1),
                       "prompt_ids_int64_le_sha256": rs[0]["prompt_ids_int64_le_sha256"],
                       "actor_rows": [r["actor_row"] for r in rs],
                       "probability_separation": {k: separation(rs, k) for k in keys}})
    def paired(rs):
        return {"rows": len(rs),
                "factual_probability_change_swap_minus_original": describe([r["swapped_factual"] - r["original_factual"] for r in rs]),
                "fullEOS_probability_change_swap_minus_original": describe([r["swapped_fullEOS"] - r["original_fullEOS"] for r in rs]),
                "original_success_root_d": describe([r["original_success_root_d"] for r in rs]),
                "swapped_success_root_d": describe([r["swapped_success_root_d"] for r in rs]),
                "success_semantic_root_sign_flips": sum(r["original_success_root_d"] * r["swapped_success_root_d"] < 0 for r in rs),
                "actual_observed_category_root_sign_flips": sum(r["original_observed_root_d"] * r["swapped_observed_root_d"] < 0 for r in rs)}
    report = {"scope": "Read-only saved native64 categorical readout task-group and paired label response; no new model/DT calls.",
              "created_unix": time.time(), "analyzer": source(Path(__file__)),
              "sources": {k: source(p) for k, p in paths.items()},
              "actual_imports": {k: data[k]["sources"] for k in ("native_rank0", "native_rank1", "label_rank0", "label_rank1")},
              "group_contract": {"rows": 64, "official_prompt_uid_groups": len(groups), "trajectories_per_group": [len(x) for x in grouped.values()],
                                 "successful_rows": sum(r["G"] for r in rows), "failure_rows": sum(r["G"] == 0 for r in rows),
                                 "group_success_counts": [sum(r["G"] for r in x) for x in grouped.values()],
                                 "group_source": "Original DataProto uid rows; exact original prefix IDs within each UID and exact first response IDs to native handler.",
                                 "source_lines": {"analyze_saved_native_b4_denominators.py": [156, 157, 222, 223, 224, 225]},
                                 "actor_B4_subdivisions_are_not_task_groups": True,
                                 "native_model_logit_dtypes": sorted({native[r["traj_uid"]]["native_logits_dtype"] for r in rows}),
                                 "label_model_logit_dtypes": sorted({label[r["traj_uid"]]["native_logits_dtype"] for r in rows})},
              "probability_definition": "exp(saved FP32 two-class categorical log-prob), semantic G=1 column1 originally/column0 after swapping; G is actual official future episode return.",
              "pooled_64": {k: separation(rows, k) for k in keys}, "actual_prompt_groups": groups,
              "within_prompt_aggregate": {k: {"success_failure_pairs": sum(g["probability_separation"][k]["success_failure_pairs"] for g in groups),
                                                "pair_weighted_AUC": sum(g["probability_separation"][k]["pairwise_wins_with_half_ties"] for g in groups) / sum(g["probability_separation"][k]["success_failure_pairs"] for g in groups),
                                                "equal_group_mean_covariance_probability_reward": sum(g["probability_separation"][k]["population_covariance_probability_reward"] for g in groups) / len(groups)} for k in keys},
              "paired_label_response": {"all64": paired(rows), "successful21": paired([r for r in rows if r["G"] == 1]), "failure43": paired([r for r in rows if r["G"] == 0])},
              "label_contract": {"same_original_prefix_current_action_world_outcome": True, "same_query_length": True, "only_two_category_label_token_IDs_swapped": True,
                                 "observed_query_source": data["label_rank0"]["sources"]["query_encoder"],
                                 "observer_source_lines": [45, 51, 52, 54, 129, 131, 133, 134, 139, 155],
                                 "root_sign_flip_scope": "Success-semantic full-response factual minus EOS log-prob; separate actual observed G-category count above. Not token-credit or policy-gradient sign."},
              "native_v2_vs_paired_original_factual_probability_difference": describe([r["original_factual"] - r["native_v2_factual"] for r in rows]),
              "rows": rows, "operations": {"local_stdlib_JSON_analysis": True, "model_calls": 0, "DT_calls": 0, "backward": 0, "updates": 0, "production_changes": 0},
              "limits": ["Eight shared official task prompts, eight sampled trajectories per prompt; rows and within-prompt pairs are dependent, not 64 independent tasks or 103 independent pairs.",
                         "Observed G is one factual outcome per trajectory, not the true conditional reward probability or a world counterfactual reference.",
                         "AUC/covariance describe reward separation in this saved batch; no acceptance threshold or probability calibration/advantage normalization is introduced.",
                         "Label swapping measures native readout encoding sensitivity for the same semantic variable; neither mapping is proposed as a repair or an oracle.",
                         "Full-response EOS root effects and readout probabilities do not establish per-token attribution, task-gradient direction, historical degradation cause, or all of the 1/211 magnitude gap.",
                         "These raw jobs used frozen query228afbc7, not the later prepared-only query-clock repair94a7; no single-token model diagnostic is used here."]}
    assert len(groups) == 8 and sum(r["G"] for r in rows) == 21
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": source(args.output), "groups": len(groups), "rows": len(rows),
                      "original_within_AUC": report["within_prompt_aggregate"]["original_factual"]["pair_weighted_AUC"],
                      "swapped_within_AUC": report["within_prompt_aggregate"]["swapped_factual"]["pair_weighted_AUC"],
                      "paired_label_response": report["paired_label_response"]["all64"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
