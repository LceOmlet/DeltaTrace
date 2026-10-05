"""Extract saved first-response attribute endpoints; never run a model."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def id_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    request_path = out / "readout-mapped-requests.json"
    case_path = out / "readout-first-response-cases.json"
    mapping_path = out / "native-minibatch-readout-mapping.json"
    mapping = json.loads(mapping_path.read_bytes())
    if identity(request_path) != mapping["mapped_requests_artifact"]:
        raise ValueError("Original mapped request asset changed")
    if identity(case_path) != mapping["case_pack"]["artifact"]:
        raise ValueError("Original first-response case asset changed")
    requests = json.loads(request_path.read_bytes())["requests"]
    pack = json.loads(case_path.read_bytes())
    cases = {(case["traj_uid"], case["source_step"]): case
             for rank_cases in pack["rank_cases"] for case in rank_cases}
    groups = defaultdict(list)
    for request in requests:
        if request["env_step"] == 0:
            groups[(request["traj_uid"], request["env_step"])].append(request)
    endpoints = []
    for (uid, step), slots in sorted(groups.items()):
        first = slots[0]
        case = cases[(uid, step)]
        ids = first["selected_input_ids"]
        if any(slot["selected_input_ids"] != ids for slot in slots):
            raise ValueError("Repeated endpoint slots do not share literal original IDs")
        if case["selected_input_ids"] != ids:
            raise ValueError("Native reader case does not match the saved original first prefix")
        score_fields = ("factual_target_logp", "reference_target_logp", "root_effect", "signed_sum")
        pooled = {field: {"mean": sum(slot["trace"][field] for slot in slots) / len(slots),
                          "min": min(slot["trace"][field] for slot in slots),
                          "max": max(slot["trace"][field] for slot in slots)}
                  for field in score_fields}
        endpoints.append({"traj_uid": uid, "source_step": step, "observed_return": case["observed_return"],
            "observed_class_index": case["observed_class_index"], "outcome_token_ids": case["outcome_token_ids"],
            "source_start": first["source_start"], "source_end": first["source_end"],
            "original_selected_input_tokens_including_target": len(ids),
            "original_selected_input_ids_sha256": id_hash(ids),
            "original_reader_input_ids_sha256": id_hash(ids[:-1]),
            "original_full_response_EOS_selected_ids_sha256": id_hash(first["reference_input_ids"]),
            "id_hash_encoding": "Canonical JSON array of original integer IDs, separators=(',',':'), UTF8; includes target except reader_input; excludes native batch EOS extension",
            "native_first_case_literal_ID_match": True,
            "transport_slots": len(slots), "pooled_duplicate_scores": pooled,
            "original_endpoint_slots": [{"rank": slot["rank"], "request_index": slot["request_index"],
                "native_report_log_line": slot["report_log_line"], "native_worker_pid": slot["worker_pid"],
                "native_compute_tokens": slot["trace"]["compute_tokens"],
                "native_dense_EOS_extension": slot["dense_eos_extension"],
                **{field: slot["trace"][field] for field in score_fields}} for slot in slots]})
    result = {"scope": __doc__, "sources": {"script": identity(Path(__file__).resolve()),
        "requests": identity(request_path), "first_response_cases": identity(case_path), "mapping": identity(mapping_path),
        "native_log": mapping["inputs"]["diagnostic_log"]},
        "coverage": {"first_response_UIDs_with_existing_attribute_endpoints": len(endpoints),
                     "transport_slots": sum(e["transport_slots"] for e in endpoints),
                     "all_native_first_cases": len(cases),
                     "G0_cases_without_original_attribute_endpoints": sum(c["observed_return"] == 0 for c in cases.values())},
        "pooling_scope": "Mean/min/max within literal-identical duplicate transport slots only; no pooling across UIDs or source steps",
        "limits": ["These endpoint scores came from original attribute paired layout with its recorded EOS extension; the native reader uses the same literal prefix without extension. Keep layout differences explicit.",
            "These are full-response deletion scores, not single-token effects or ground-truth world probabilities.",
            "The43 G0 cases did not receive original DT readout; no scores are invented for them."],
        "endpoints": endpoints}
    path = out / "native64-first-response-existing-endpoints.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": identity(path), "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
