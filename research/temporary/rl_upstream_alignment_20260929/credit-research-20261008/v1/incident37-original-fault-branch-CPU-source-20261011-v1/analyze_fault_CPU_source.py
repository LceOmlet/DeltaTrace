"""Read saved failure-time native CPU sources and original GEMM operands.

Diagnostic only: no model, DT, optimizer, GPU call, or numerical acceptance.
Use exact saved dtype comparisons; report missing observations as missing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def summary(value, normal, bad):
    value = value.contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
        min=float(value.min()), max=float(value.max()), L2=float(value.double().norm()),
        sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        exact_normal_checkpoint_row=torch.equal(value, normal),
        exact_actual_GEMM_row=torch.equal(value, bad),
        maxabs_to_normal=float((value.float()-normal.float()).abs().max()),
        maxabs_to_actual_GEMM=float((value.float()-bad.float()).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    root = args.job.parents[1]
    result = dict(unix=time.time(), source=identity(Path(__file__)),
        ranks=[], GPU_calls=0, model_calls=0, DT_calls=0,
        optimizer_steps=0, production_changes=0, new_tolerance=None)
    for rank in range(2):
        path = args.job / "result" / f"rank{rank}-original-fault-CPU-source.pt"
        row = dict(rank=rank, path=str(path))
        result["ranks"].append(row)
        if not path.is_file():
            row["missing"] = True
            continue
        source = load(path)
        gemm = load(args.job / "result" / f"rank{rank}-native-GEMM-rows.pt")
        prior = load(root / "receipts" / "textcraft-native-incident37-native-head-live-capture-20261011-v1" /
                     "result" / f"rank{rank}-native-head-live-microbatch1.pt")
        normal = prior["tensors"]["vocab_weights"][159048]
        bad = gemm["competing_weights"][gemm["metadata"]["copied_indices"][0]]
        row.update(artifact=identity(path), errors=source["errors"],
                   microbatch=source["microbatch"], parameters=[])
        for parameter in source["parameters"]:
            item = {k:v for k,v in parameter.items() if not isinstance(v,torch.Tensor)}
            row["parameters"].append(item)
            for key in ["actual_flat_source", "sharded_local_tensor"]:
                if key in parameter:
                    item[key] = summary(parameter[key], normal, bad)
        row["actual_GEMM_row"] = summary(bad, normal, bad)
    result.update(seconds=time.perf_counter()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        CUDA_initialized=torch.cuda.is_initialized())
    out = args.job / "fault-CPU-source-analysis.json"
    out.write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
