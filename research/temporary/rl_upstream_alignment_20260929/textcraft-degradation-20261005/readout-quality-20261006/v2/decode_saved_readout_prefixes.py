"""Decode original saved native64 IDs with the original offline tokenizer on CPU.

No model/weights, generation, probability prediction, credit or update.  The
original arrays are split only at their already saved source/query boundaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def resources(psutil):
    memory = psutil.Process().memory_full_info()
    return {"rss_bytes": memory.rss, "pss_bytes": getattr(memory, "pss", None),
            "host_available_bytes": psutil.virtual_memory().available}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_out", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Decode only in the original CPU environment with CUDA hidden")
    import psutil
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise ValueError("No GPU or process group may be initialized")
    before = resources(psutil)
    mapping_path = args.native_out / "native-minibatch-readout-mapping.json"
    mapping = json.loads(mapping_path.read_bytes())
    config = mapping["readout_config"]
    pack_record = identity(mapping["case_pack"]["artifact"]["path"])
    expected = mapping["case_pack"]["artifact"]
    if pack_record["sha256"] != expected["sha256"] or pack_record["bytes"] != expected["bytes"]:
        raise ValueError("Original saved case pack changed")
    if pack_record["sha256"] != "3ca15c0ac56b0d73fb3b5948330773d95ffb1990e6fe0edca53fbf5b67575d41":
        raise ValueError("Not the requested original saved native64 case pack")
    pack = json.loads(Path(pack_record["path"]).read_bytes())
    boundary_path = args.native_out / "native-minibatch-update-boundary.json"
    boundary = json.loads(boundary_path.read_bytes())
    tokenizer_path = Path(config["model_tokenizer_path"])
    tokenizer_files = [identity(tokenizer_path / name) for name in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json") if (tokenizer_path / name).is_file()]
    # Exactly the producer's existing tokenizer API; no model class is loaded.
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    decode = lambda values: tokenizer.decode(values, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
    records = []
    groups = defaultdict(list)
    query_hashes = set()
    prompt_hashes = set()
    for rank, cases in enumerate(pack["rank_cases"]):
        for index, case in enumerate(cases):
            row = rank * 32 + index
            if case["actor_row"] != row or case["traj_uid"] != boundary["traj_uid"][row]:
                raise ValueError("Original actor-row/UID order differs from the saved boundary")
            ids = case["selected_input_ids"]
            start, end = case["source_start"], case["source_end"]
            prompt, actions, query = ids[:start], ids[start:end], ids[end:-1]
            if len(query) != case["query_tokens"] or len(actions) != case["retained_source_tokens"]:
                raise ValueError("Saved source/query split does not bind the original arrays")
            if case["source_step"] != 0 or case["outcome_token_ids"] != [15, 16]:
                raise ValueError("Not the recorded initial native64 outcome reader")
            if ids[-1] != case["target_id"] or ids[-1] != [15, 16][case["observed_class_index"]]:
                raise ValueError("Recorded target does not select the original observed category")
            query_hash = ids_hash(query)
            prompt_hash = ids_hash(prompt)
            prompt_hashes.add(prompt_hash)
            query_hashes.add(query_hash)
            record = {"rank": rank, "case_index": index, "actor_row": row,
                      "prompt_uid": boundary["uid"][row], "traj_uid": case["traj_uid"],
                      "source_step": case["source_step"], "observed_return": case["observed_return"],
                      "source_start": start, "source_end": end, "context_tokens": len(ids),
                      "source_token_count": len(actions), "query_token_count": len(query),
                      "target_id": ids[-1], "target_decode": decode([ids[-1]]),
                      "original_metadata": case["original_metadata"],
                      "selected_input_ids_sha256": ids_hash(ids), "prompt_ids_sha256": prompt_hash,
                      "query_ids_sha256": query_hash, "prompt_decode": decode(prompt),
                      "current_response_decode": decode(actions), "query_decode": decode(query)}
            records.append(record)
            groups[boundary["uid"][row]].append(record)
    unique_prompts = []
    for uid, cases in groups.items():
        hashes = {case["prompt_ids_sha256"] for case in cases}
        unique_prompts.append({"prompt_uid": uid, "cases": len(cases),
            "distinct_original_prompt_ID_arrays": len(hashes),
            "prompt_ids_sha256": cases[0]["prompt_ids_sha256"],
            "prompt_decode": cases[0]["prompt_decode"],
            "actor_rows": [case["actor_row"] for case in cases]})
    result = {"scope": "CPU decode-only observation of the original saved native64 first-response ID arrays; no new template, target or state constructed.",
        "inputs": {"case_pack": pack_record, "mapping": identity(mapping_path),
                   "update_boundary": identity(boundary_path)},
        "sources": {"decoder": identity(__file__), "tokenizer_factory": identity(inspect.getsourcefile(AutoTokenizer)),
                    "tokenizer_decode": identity(inspect.getsourcefile(type(tokenizer).decode)),
                    "tokenizer_files": tokenizer_files, "torch": identity(torch.__file__)},
        "tokenizer": {"path": str(tokenizer_path), "class": type(tokenizer).__name__,
            "local_files_only": True, "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False, "vocab_size": tokenizer.vocab_size,
            "outcome_ids": [15, 16], "outcome_decodes": [decode([15]), decode([16])]},
        "coverage": {"cases": len(records), "prompt_UID_groups": len(groups),
            "distinct_prompt_ID_arrays": len(prompt_hashes), "distinct_query_ID_arrays": len(query_hashes),
            "source_steps": sorted({case["source_step"] for case in records}),
            "max_context_tokens": max(case["context_tokens"] for case in records),
            "recorded_max_length": config["max_length"], "recorded_max_steps": config["max_steps"]},
        "original_prompts": unique_prompts, "cases": records,
        "runtime": {"pid": os.getpid(), "pid_birth": psutil.Process().create_time(),
            "python": os.sys.executable, "torch_version": torch.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_initialized": torch.cuda.is_initialized(), "distributed_initialized": torch.distributed.is_initialized(),
            "started_unix": started, "elapsed_seconds": time.time() - started,
            "before": before, "after": resources(psutil),
            "maxRSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "operations": {"model_loads": 0, "model_forwards": 0, "probability_predictions": 0,
                       "backward_calls": 0, "optimizer_steps": 0},
        "limits": ["Decoded text is inspection of original IDs, not a new environment/action parser or a re-encoded actor input.",
                   "Initial-context decoding covers64 first responses only; later186 requests retain their source audit and saved probability scope.",
                   "A generic forecast query or fixed label assignment alone cannot identify causal label-word bias."]}
    if result["runtime"]["cuda_initialized"] or result["runtime"]["distributed_initialized"]:
        raise ValueError("CPU-only decode contract failed")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": identity(args.output), "coverage": result["coverage"],
                      "tokenizer": result["tokenizer"], "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
