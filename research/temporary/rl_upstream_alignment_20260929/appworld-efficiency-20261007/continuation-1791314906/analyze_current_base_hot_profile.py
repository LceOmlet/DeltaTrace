"""Read original completed B8 profiler traces; no torch/model or runtime changes."""
import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--analyzer", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    spec = importlib.util.spec_from_file_location("original_trace_analyzer", args.analyzer)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result_path = root / "result.json"
    if not result_path.exists():
        print(json.dumps({"completed": False, "root": str(root)}))
        return
    prepared = json.loads((root / "prepared.json").read_bytes())
    result = {"scope": "Original real B8 current-base hot trace; instrumented GPU event sums may overlap and are not wall-time savings or numerical acceptance.",
        "completed_result": {"path": str(result_path), "sha256": sha(result_path)},
        "prepared": {"path": str(root / "prepared.json"), "sha256": sha(root / "prepared.json")},
        "current_formal_source": prepared["current_formal_owner"],
        "checkpoint": prepared["checkpoint"], "base_model_initialization": prepared["base_model_initialization"],
        "analyzer": {"path": str(args.analyzer), "sha256": sha(args.analyzer)},
        "source": {"path": __file__, "sha256": sha(Path(__file__))}, "ranks": []}
    for rank in (0, 1):
        path = root / f"rank{rank}.json"
        raw = json.loads(path.read_bytes())
        trace = Path(raw["trace"]["path"])
        assert sha(trace) == raw["trace"]["sha256"]
        analysis = owner.analyze(trace, parameter_ranges=True)
        data = json.loads(trace.read_bytes())
        kernels, copies = defaultdict(lambda: [0, 0.]), defaultdict(lambda: [0, 0, 0.])
        for event in data["traceEvents"]:
            if event.get("ph") != "X": continue
            category, name = event.get("cat"), event.get("name", "")
            if category == "kernel":
                row = kernels[name]; row[0] += 1; row[1] += event.get("dur", 0) / 1e6
            elif category == "gpu_memcpy":
                byte_count = event.get("args", {}).get("bytes")
                direction = next((v for v in ("HtoD", "DtoH", "DtoD") if v in name), name)
                row = copies[(direction, byte_count)]; row[0] += 1; row[1] += byte_count or 0; row[2] += event.get("dur", 0) / 1e6
        result["ranks"].append({"rank": rank, "rank_source": {"path": str(path), "sha256": sha(path)},
            "context_lengths": raw["context_lengths"], "native_conv_initial_states": raw["native_conv_initial_states"],
            "restored_checkpoint": raw["restored_checkpoint"], "imported_sources": raw["imported_sources"],
            "reports": {k: {"wall_seconds": v["total_wall_seconds"], "attribute_seconds": v["original_attribute_wall_seconds"],
                "phase_seconds": v["original_runner_phase_seconds"], "phase_counts": v["original_runner_phase_counts"],
                "physical_free_bytes": v["physical_free_bytes"], "pss_bytes": v["pss_bytes"]} for k, v in raw["reports"].items()},
            "trace_analysis": analysis,
            "top_kernel_names": sorted([{"name": k, "count": v[0], "device_seconds": v[1]} for k, v in kernels.items()], key=lambda v: -v["device_seconds"])[:35],
            "copies_by_exact_size": sorted([{"direction": k[0], "bytes_per_event": k[1], "count": v[0], "total_bytes": v[1], "device_seconds": v[2]} for k, v in copies.items()], key=lambda v: -v["device_seconds"])})
    target = root / "read-only-current-base-profile-analysis.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "receipt": str(target), "sha256": sha(target),
        "ranks": [{"rank": r["rank"], "reports": {k: v["wall_seconds"] for k,v in r["reports"].items()},
            "copies": r["trace_analysis"]["copies_by_owner"], "kernels": r["trace_analysis"]["kernels_by_owner"],
            "parameter_cpu": r["trace_analysis"]["parameter_cpu_by_stage"],
            "parameter_device": r["trace_analysis"]["parameter_device_by_stage"],
            "top_kernel_names": r["top_kernel_names"][:8], "exact_copies": r["copies_by_exact_size"][:8]} for r in result["ranks"]]}))


if __name__ == "__main__": main()
