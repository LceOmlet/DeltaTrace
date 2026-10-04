"""Read original Chrome trace device events; never import torch or alter a run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

NATIVE = re.compile(r"^DT_native_layer_(\d+)_pass_(\d+)$")
SIZE_BINS = ((1024, "<1KiB"), (2**20, "1KiB..<1MiB"),
             (16*2**20, "1MiB..<16MiB"), (64*2**20, "16MiB..<64MiB"),
             (256*2**20, "64MiB..<256MiB"), (2**30, "256MiB..<1GiB"))


def external_id(event):
    value = event.get("args", {}).get("External id")
    return None if value is None else str(value)


def duration(event):
    return float(event.get("dur", 0))


def size_bin(value):
    if value is None:
        return "unknown"
    for upper, label in SIZE_BINS:
        if value < upper:
            return label
    return ">=1GiB"


def family(name, cpu_name):
    if cpu_name in ("aten::mm", "aten::addmm", "aten::bmm"):
        return cpu_name.removeprefix("aten::")
    combined = (name + " " + cpu_name).lower().replace("_", "")
    if "allgather" in combined:
        return "all_gather"
    if "allreduce" in combined:
        return "all_reduce"
    return "other_kernel"


def analyze(path, root_pass=1, replay_pass=2, top=25):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    cpu_by_id = defaultdict(list)
    ranges = defaultdict(list)
    pass_counts = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category == "cpu_op" and external_id(event) is not None:
            cpu_by_id[external_id(event)].append(event)
        matched = NATIVE.fullmatch(event.get("name", ""))
        if matched:
            layer, pass_number = map(int, matched.groups())
            ranges[(event.get("pid"), event.get("tid"))].append(event)
            pass_counts[pass_number] += 1
    for regions in ranges.values():
        regions.sort(key=lambda e: (float(e["ts"]), duration(e)))

    tables = {key: defaultdict(lambda: [0, 0, 0, 0.0]) for key in
              ("copies_by_owner", "copies_by_layer", "copies_by_cpu_op", "copies_by_size_bin",
               "kernels_by_owner", "kernels_by_layer", "mm_by_owner", "collectives_by_owner")}
    links = Counter()
    unknown_examples = []
    raw_counts = Counter()
    total_device_us = 0.0

    def add(table, key, event, byte_count=None):
        values = tables[table][key]
        values[0] += 1
        values[1] += byte_count or 0
        values[2] += int(byte_count is None)
        values[3] += duration(event)

    for event in events:
        # Only actual GPU kernel/memcpy records. Never include cpu_op,
        # gpu_annotation, key_averages, flow links, or their nested totals.
        category = event.get("cat", "")
        if event.get("ph") != "X" or category not in ("kernel", "gpu_memcpy"):
            continue
        raw_counts[category] += 1
        total_device_us += duration(event)
        candidates = cpu_by_id.get(external_id(event), [])
        cpu = candidates[0] if len(candidates) == 1 else None
        if cpu is None:
            reason = "ambiguous_cpu_id" if candidates else "missing_cpu_id"
            links[reason] += 1
            cpu_name, owner, layer_name = "unknown_cpu_op", "unknown_owner", "unknown_owner"
            if len(unknown_examples) < top:
                unknown_examples.append({"name": event.get("name"), "external_id": external_id(event),
                                         "reason": reason, "duration_us": duration(event)})
        else:
            links["linked_cpu_id"] += 1
            cpu_name = cpu.get("name", "unknown_cpu_op")
            # Attribute using the CPU launch op's own thread/time, never the
            # later asynchronous GPU timestamp. Choose the innermost range.
            start = float(cpu["ts"])
            containing = [r for r in ranges.get((cpu.get("pid"), cpu.get("tid")), [])
                          if float(r["ts"]) <= start < float(r["ts"]) + duration(r)]
            if containing:
                region = min(containing, key=lambda r: (duration(r), -float(r["ts"])))
                layer, pass_number = map(int, NATIVE.fullmatch(region["name"]).groups())
                owner = ("root" if pass_number == root_pass else
                         "replay" if pass_number == replay_pass else f"native_pass_{pass_number}")
                layer_name = f"{owner}:layer_{layer}"
                links["inside_native_range"] += 1
            else:
                owner = layer_name = "outside_native"
                links["outside_native_range"] += 1
        if category == "gpu_memcpy":
            name = event.get("name", "")
            direction = next((kind for kind in ("HtoD", "DtoH", "DtoD") if kind in name), "other_memcpy")
            raw_bytes = event.get("args", {}).get("bytes")
            byte_count = int(raw_bytes) if raw_bytes is not None else None
            add("copies_by_owner", (direction, owner), event, byte_count)
            add("copies_by_layer", (direction, layer_name), event, byte_count)
            add("copies_by_cpu_op", (direction, owner, cpu_name), event, byte_count)
            add("copies_by_size_bin", (direction, owner, size_bin(byte_count)), event, byte_count)
        else:
            kind = family(event.get("name", ""), cpu_name)
            add("kernels_by_owner", (kind, owner), event)
            add("kernels_by_layer", (kind, layer_name), event)
            if kind in ("mm", "addmm", "bmm"):
                add("mm_by_owner", (kind, owner, cpu_name), event)
            elif kind in ("all_gather", "all_reduce"):
                add("collectives_by_owner", (kind, owner, cpu_name), event)

    result = {
        "source_path": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_trace_source": data.get("source") if isinstance(data, dict) else None,
        "trace_event_count": len(events), "raw_device_event_counts": dict(raw_counts),
        "raw_device_duration_seconds_sum": total_device_us / 1e6,
        "link_counts": dict(links), "native_range_pass_counts": dict(sorted(pass_counts.items())),
        "owner_pass_mapping": {"root": root_pass, "replay": replay_pass},
        "measurement_scope": "Sum each original ph=X kernel/gpu_memcpy duration once; overlapping streams may overlap, so sums are not wall time. Bytes only come from original memcpy args; no tensor-shape estimates.",
        "owner_rule": "GPU External id to one original cpu_op with same id, then innermost DT_native_layer range containing CPU op start on same pid/tid. Missing/ambiguous ids remain unknown, not guessed.",
        "mapping_scope": "Pass1=root and pass2=replay apply to hot-shared_warm whose bank preparation is outside attribute. Other pass values are reported verbatim.",
        "unknown_link_examples": unknown_examples,
    }
    for table, values in tables.items():
        rows = [{"group": list(key), "count": value[0], "known_bytes": value[1],
                 "events_without_bytes": value[2], "device_seconds": value[3]/1e6}
                for key, value in values.items()]
        result[table] = sorted(rows, key=lambda row: -row["device_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-pass", type=int, default=1)
    parser.add_argument("--replay-pass", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    results = [analyze(path, args.root_pass, args.replay_pass, args.top) for path in args.traces]
    encoded = json.dumps({"traces": results}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "traces": len(results)}))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
