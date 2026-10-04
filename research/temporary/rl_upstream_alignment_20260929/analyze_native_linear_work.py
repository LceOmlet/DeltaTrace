"""CPU-only attribution of saved native hot-profile linear work.

Reads the existing CPU/device links; it does not import Torch, execute a model,
change an operator, or define a numerical acceptance threshold. The signature
checks describe this one recorded owner workload, not a generic profiler rule.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[3]
DEFAULT_LINKS = REPO / "research/temporary/rl_upstream_alignment_20260929/phase-observation-20261004/native-mm-copy-cpu-device-links.json"
DEFAULT_OUTPUT = REPO / "experiments/rl/results_native_linear_work_20261004.json"
SOURCE_REFERENCES = {
    "DT_linear_owner": ("deltatrace/clean/qwen35/qwen35_decoder_finite.py", [[46, 79], [95, 98], [192, 217], [228, 230]]),
    "DT_GDN_owner": ("deltatrace/clean/qwen35/qwen35_gdn_finite.py", [[215, 244], [283, 287]]),
    "DT_vendor_mm": ("deltatrace/clean/qwen35/finite_fla_gpu.py", [[90, 93]]),
    "PEFT_owner": ("research/temporary/rl_upstream_alignment_20260929/native-torch-sac-20261004/peft_lora_layer.py", [[745, 777], [779, 820]]),
    "Qwen_native_owner": ("research/temporary/reward_readout_20260922/prefix_reuse/modeling_qwen3_5.py", [[151, 169], [1823, 1827]]),
    "DT_categorical_head": ("deltatrace/clean/qwen35/qwen35_answer_finite.py", [[86, 110]]),
}
REPLAY_NAME = re.compile(r"DT_native_layer_(\d+)_pass_2")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_op(op: dict) -> dict:
    return {
        "cpu_name": op["cpu_name"],
        "cpu_external_id": op["args"]["External id"],
        "cpu_ts_microseconds": op["cpu_ts"],
        "cpu_pid": op["cpu_pid"],
        "cpu_tid": op["cpu_tid"],
        "linked_device_event_count": op["device_events"],
        "linked_device_seconds_sum": op["device_seconds"],
    }


def inside(op: dict, annotation: dict) -> bool:
    return (op["cpu_pid"] == annotation["pid"]
            and op["cpu_tid"] == annotation["tid"]
            and annotation["ts"] <= op["cpu_ts"] < annotation["ts"] + annotation["dur"])


def total(ops: list[dict]) -> dict:
    return {
        "cpu_call_count": len(ops),
        "linked_device_event_count": sum(op["device_events"] for op in ops),
        "linked_device_seconds_sum": sum(op["device_seconds"] for op in ops),
    }


def analyze_rank(record: dict) -> dict:
    annotations = record["annotations"]
    replay = sorted((a for a in annotations if REPLAY_NAME.fullmatch(a["name"])), key=lambda a: a["ts"])
    layers = [int(REPLAY_NAME.fullmatch(a["name"])[1]) for a in replay]
    if layers != list(range(31, -1, -1)):
        raise ValueError("This receipt requires the recorded complete reverse 31..0 owner pass")
    by_thread = {}
    for op in record["cpu_ops"]:
        if op["cpu_name"] in ("aten::mm", "aten::bmm"):
            by_thread.setdefault((op["cpu_pid"], op["cpu_tid"]), []).append(op)
    for ops in by_thread.values():
        ops.sort(key=lambda op: op["cpu_ts"])
    groups = {name: [] for name in ("dense_adapter_materialization_mm", "base_linear_bmm",
                                   "dense_adapter_linear_bmm", "MLP_dense_adapter_linear_bmm", "FLA_bmm")}
    matched_mm = set()
    matched_bmm = set()
    windows = []
    family_counts = Counter()
    for index, annotation in enumerate(replay):
        thread_ops = by_thread[(annotation["pid"], annotation["tid"])]
        times = [op["cpu_ts"] for op in thread_ops]
        start = annotation["ts"] + annotation["dur"]
        end = replay[index + 1]["ts"] if index + 1 < len(replay) else None
        begin_index = bisect_left(times, start)
        end_index = bisect_left(times, end) if end is not None else len(times)
        selected = thread_ops[begin_index:end_index]
        mm = [op for op in selected if op["cpu_name"] == "aten::mm"]
        bmm = [op for op in selected if op["cpu_name"] == "aten::bmm"]
        # The unchanged finite-owner source has three MLP linear maps first.
        # FA adds o/q/k/v; GDN adds out/qkv/z/b/a. Each active map performs
        # adjacent base+delta _mm calls. The GDN FLA callback is between the
        # first four and last four linear maps. These exact recorded signatures
        # are checked before assigning costs; they are not numerical tolerances.
        if (len(mm), len(bmm)) == (7, 14):
            family = "FA_signature"
            names = ["mlp.down", "mlp.up", "mlp.gate", "attention.o", "attention.q", "attention.k", "attention.v"]
            linear = bmm
            fla = []
        elif (len(mm), len(bmm)) == (8, 160):
            family = "GDN_signature"
            names = ["mlp.down", "mlp.up", "mlp.gate", "gdn.out", "gdn.qkv", "gdn.z", "gdn.b", "gdn.a"]
            linear = bmm[:8] + bmm[-8:]
            fla = bmm[8:-8]
        else:
            raise ValueError(f"Unmatched recorded layer {layers[index]} mm/bmm signature: {len(mm)}/{len(bmm)}")
        family_counts[family] += 1
        base, delta = linear[::2], linear[1::2]
        groups["dense_adapter_materialization_mm"].extend(mm)
        groups["base_linear_bmm"].extend(base)
        groups["dense_adapter_linear_bmm"].extend(delta)
        groups["MLP_dense_adapter_linear_bmm"].extend(delta[:3])
        groups["FLA_bmm"].extend(fla)
        matched_mm.update(op["args"]["External id"] for op in mm)
        matched_bmm.update(op["args"]["External id"] for op in bmm)
        windows.append({
            "layer": layers[index], "family_from_recorded_owner_signature": family,
            "cpu_pid": annotation["pid"], "cpu_tid": annotation["tid"],
            "after_annotation": annotation["name"], "start_cpu_microseconds": start,
            "before_annotation": replay[index + 1]["name"] if end is not None else None,
            "end_cpu_microseconds": end,
            "materialization_mm": [compact_op(op) for op in mm],
            "linear_pairs_in_owner_call_order": [
                {"projection": name, "base": compact_op(base_op), "dense_adapter": compact_op(delta_op)}
                for name, base_op, delta_op in zip(names, base, delta)],
            "FLA_bmm": {**total(fla), "cpu_external_ids": [op["args"]["External id"] for op in fla]},
        })
    all_mm = [op for op in record["cpu_ops"] if op["cpu_name"] == "aten::mm"]
    all_bmm = [op for op in record["cpu_ops"] if op["cpu_name"] == "aten::bmm"]
    native = [a for a in annotations if a["name"].startswith("DT_native_layer_")]
    native_mm = [op for op in all_mm if any(inside(op, a) for a in native)]
    remaining_mm = [op for op in all_mm if op["args"]["External id"] not in matched_mm
                    and not any(inside(op, a) for a in native)]
    remaining_bmm = [op for op in all_bmm if op["args"]["External id"] not in matched_bmm]
    if (len(native_mm), len(remaining_mm), len(remaining_bmm)) != (1488, 3, 1):
        raise ValueError("Original CPU-call count did not close against the recorded native/head boundaries")
    first_root = next(a for a in annotations if a["name"] == "DT_native_layer_0_pass_1")
    last_root = next(a for a in annotations if a["name"] == "DT_native_layer_31_pass_1")
    if not (remaining_bmm[0]["cpu_ts"] < first_root["ts"]):
        raise ValueError("The remaining BMM no longer has the recorded pre-decoder location")
    remaining_mm.sort(key=lambda op: op["cpu_ts"])
    if not all(last_root["ts"] + last_root["dur"] <= op["cpu_ts"] < replay[0]["ts"] for op in remaining_mm):
        raise ValueError("The three remaining MM calls are no longer at the recorded head/seed boundary")
    summaries = {name: total(ops) for name, ops in groups.items()}
    if [summaries[name]["cpu_call_count"] for name in ("dense_adapter_materialization_mm", "base_linear_bmm",
            "dense_adapter_linear_bmm", "MLP_dense_adapter_linear_bmm", "FLA_bmm")] != [248, 248, 248, 96, 3456]:
        raise ValueError("Recorded finite linear/FLA call counts did not close")
    return {
        "rank": record["rank"], "original_trace": record["source"],
        "scope": "One saved shared_warm attribute, original B4 per rank; bank preparation excluded",
        "all_cpu_mm": total(all_mm), "all_cpu_bmm": total(all_bmm),
        "native_forward_and_replay_mm": total(native_mm), "finite_groups": summaries,
        "family_signature_counts": dict(family_counts),
        "head_boundary_mm": [
            {"source_order_inference": name, **compact_op(op)}
            for name, op in zip(("native_full_vocabulary_head", "categorical_FP32_head", "categorical_seed_transpose"), remaining_mm)],
        "pre_decoder_bmm": {
            "classification": "pre-decoder BMM, not head",
            "source_supported_inference": "Qwen rotary_emb FP32 inv_freq @ position_ids; shapes were not recorded",
            "before_annotation": first_root["name"], **compact_op(remaining_bmm[0]),
        },
        "count_closure": {
            "bmm": "3953 = 248 base + 248 dense adapter + 3456 FLA + 1 pre-decoder",
            "mm": "1739 = 1488 native layer forward/replay + 248 dense adapter materialization + 3 head/seed",
            "head_detail": "The native full-vocabulary head is one MM, not the remaining pre-decoder BMM",
        },
        "layer_windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    links = args.links.resolve()
    data = json.loads(links.read_text(encoding="utf-8"))
    if sorted(record["rank"] for record in data) != [0, 1]:
        raise ValueError("This recorded workload has exactly the original two ranks")
    sources = {name: {"path": path, "sha256": sha256(REPO / path), "line_ranges": lines}
               for name, (path, lines) in SOURCE_REFERENCES.items()}
    output = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "role": "CPU-only analysis of saved trace; no model, GPU, production change or numerical acceptance",
        "measurement": "Sum original linked device durations. Streams can overlap; these sums are not wall time or predicted training speed.",
        "mapping": "Original CPU pid/tid/launch timestamp and External id; complete reverse-layer windows; operator order from unchanged owner source",
        "identity_limit": "Original record_shapes=False. Base/delta pairing and projection labels are owner-call-order mappings, not observed tensor identities.",
        "local_link_extract": {"path": str(links), "sha256": sha256(links), "bytes": links.stat().st_size},
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "owner_sources": sources,
        "dtype_scope": {
            "actual_adapter_compute_dtype": "Not present in this link extraction; do not infer it from base dtype or initial FP32 adapter storage",
            "dense_delta_bytes_formula": "out_features * in_features * actual get_delta_weight output element_size",
            "MLP_shape_source": "results_native_projection_inputs_20261004.json records 12288x4096/4096x12288 base weights",
            "conditional_MLP_delta_bytes": {"one_BF16_delta": 100663296, "three_BF16_deltas": 301989888,
                                           "one_FP32_delta": 201326592, "three_FP32_deltas": 603979776},
            "memory_limit": "Logical conditional bytes only; not a measured allocator/physical peak",
        },
        "ranks": [analyze_rank(record) for record in sorted(data, key=lambda record: record["rank"])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256(args.output),
                      "ranks": [{"rank": rank["rank"], "finite_groups": rank["finite_groups"]} for rank in output["ranks"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
