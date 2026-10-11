"""Diagnose actual-size native FSDP copy-out; no model or training execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback
import unittest

import torch
from safetensors import safe_open
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as owner
from torch.distributed.fsdp._fully_shard._fsdp_param import alloc_storage, free_storage


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_tensor(model, index, key):
    path = model / index["weight_map"][key]
    with safe_open(path, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(key)
    return value, dict(key=key, path=str(path), file_bytes=path.stat().st_size,
                       shape=list(value.shape), dtype=str(value.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    report = dict(unix=time.time(), source=identity(Path(__file__)),
                  installed_owner=identity(Path(owner.__file__)), torch_version=torch.__version__,
                  official_test_source=identity(args.source / "test_torch_v2.8.0.py"),
                  model_calls=0, DT_calls=0, PPO_updates=0, production_changes=0,
                  formal_restarted=False, numerical_tolerance=None,
                  property="Exact data equality, matching official test_split_with_sizes_copy_out; not a floating-point acceptance threshold.",
                  cases=[], limitation="Single-process serialized native copy/storage checks; not a distributed collective, race, whole-model, or 32k training acceptance.")
    try:
        assert report["installed_owner"]["sha256"] == "3bf079219682d1a9439b81754af65e87f69157ec69f20f065e5565519f7093d9"
        # Execute the unchanged upstream method body. It only uses assertTrue
        # and assertFalse, whose Boolean semantics unittest.TestCase supplies.
        official = ast.parse((args.source / "test_torch_v2.8.0.py").read_text())
        node = next(n for n in ast.walk(official) if isinstance(n, ast.FunctionDef)
                    and n.name == "test_split_with_sizes_copy_out")
        namespace = {"torch": torch}
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(args.source / "test_torch_v2.8.0.py"), "exec"), namespace)
        t = time.perf_counter()
        namespace[node.name](unittest.TestCase())
        report["unchanged_upstream_method"] = dict(name=node.name, passed=True,
                                                     seconds=time.perf_counter()-t,
                                                     includes_original_CUDA_graph_cases=True)
        metadata = json.loads((args.source / "root-metadata.json").read_text())
        params = metadata["record"]["parameters"]
        assert [p["orig_size"] for p in params] == [[4096], [248320, 4096]]
        splits = [p["actual_inputs"][0]["numel"] for p in params]
        assert splits == [2048, 508559360]
        assert all(p["actual_inputs"][0]["dtype"] == "torch.bfloat16" for p in params)
        report["geometry_source"] = metadata
        report["split_sizes"] = splits
        model = Path("/mnt/si0021787ci2/default/models/Qwen3.5-9B")
        index_path = model / "model.safetensors.index.json"
        index = json.loads(index_path.read_bytes())
        norm_keys = [k for k in index["weight_map"] if k.endswith(".norm.weight") and ".layers." not in k
                     and (k == "model.norm.weight" or k == "model.language_model.norm.weight")]
        assert len(norm_keys) == 1, norm_keys
        head, head_identity = load_tensor(model, index, "lm_head.weight")
        norm, norm_identity = load_tensor(model, index, norm_keys[0])
        head = head.reshape(2, splits[1])
        norm = norm.to(torch.bfloat16).reshape(2, splits[0])
        report["checkpoint_inputs"] = [head_identity, norm_identity]
        report["checkpoint_index"] = identity(index_path)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        packed = torch.empty((2, sum(splits)), dtype=torch.bfloat16, device=device)
        packed[:, :splits[0]].copy_(norm)
        packed[:, splits[0]:].copy_(head)
        outputs = [torch.empty((2, n), dtype=torch.bfloat16, device=device) for n in splits]
        for name in ("fresh_storage", "native_free_and_reallocate_storage"):
            if name != "fresh_storage":
                for tensor in outputs:
                    free_storage(tensor)
                    alloc_storage(tensor)
            # Any unwritten element remains NaN, so even originally zero
            # checkpoint elements cannot hide an incomplete copy.
            for tensor in outputs:
                tensor.fill_(float("nan"))
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.autograd._unsafe_preserve_version_counter(tuple(outputs)):
                torch.ops.fsdp.split_with_sizes_copy(packed, splits, dim=1, out=outputs)
            end.record()
            end.synchronize()
            case = dict(name=name, native_copy_ms=begin.elapsed_time(end),
                        source_pointer=packed.data_ptr(), output_pointers=[t.data_ptr() for t in outputs], fields=[])
            for source, output in zip(packed.split_with_sizes(splits, dim=1), outputs):
                equal = bool(source.eq(output).all().item())
                field = dict(shape=list(output.shape), all_elements_equal=equal)
                if not equal:
                    count, first = 0, []
                    for row in range(2):
                        for start in range(0, output.shape[1], 1 << 20):
                            actual = output[row, start:start+(1 << 20)]
                            expected = source[row, start:start+(1 << 20)]
                            bad = actual.ne(expected)
                            count += int(bad.sum().item())
                            if len(first) < 8 and bool(bad.any().item()):
                                indices = bad.nonzero().flatten()[:8-len(first)].cpu().tolist()
                                first.extend(dict(row=row, column=start+i,
                                                  actual=float(actual[i]), expected=float(expected[i])) for i in indices)
                    field.update(mismatched_elements=count, first_mismatches=first)
                case["fields"].append(field)
            known_row = outputs[1].reshape(248320, 4096)[159048].cpu()
            case["head_row159048_sha256"] = hashlib.sha256(known_row.view(torch.uint8).numpy().tobytes()).hexdigest()
            report["cases"].append(case)
            print(json.dumps(dict(phase=name, fields=case["fields"], copy_ms=case["native_copy_ms"])), flush=True)
        report["all_actual_size_copy_cases_passed"] = all(f["all_elements_equal"] for c in report["cases"] for f in c["fields"])
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["physical_while_buffers_resident"] = subprocess.run(["mx-smi"], capture_output=True, text=True, timeout=15).stdout
        report["status"] = "copy_only_checks_complete"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.perf_counter()-started
        report["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report["CUDA_initialized"] = torch.cuda.is_initialized()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k in ["status","seconds","peak_RSS_bytes","all_actual_size_copy_cases_passed","peak_torch_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
