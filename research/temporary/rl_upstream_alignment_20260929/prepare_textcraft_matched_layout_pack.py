"""Prepare original native64 B4 iterations0--6 on CPU; no model or DT run."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import psutil
import torch
from transformers import AutoTokenizer
from verl.protocol import DataProto
from agent_system.multi_turn_rollout.utils import to_list_of_dict
import analyze_textcraft_native_minibatch as stats
import map_textcraft_native64_readout as mapping_owner
import dt_training_batch
from reward_readout import EventRatioReadout
from deltatrace_credit import trace_token_attribution
from deltatrace_rollout import _Qwen35CausalOwnerView


class CapturedOwnerAttribute(Exception):
    pass


class ObservePack:
    model = None

    def attribute(self, pair, mask, selection, **kwargs):
        self.pair, self.mask, self.selection, self.kwargs = pair, mask, selection, kwargs
        raise CapturedOwnerAttribute()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    start = time.time()
    torch.set_num_threads(1)
    completed = json.loads((out / "native-minibatch-readout-mapping.json").read_bytes())
    launch = json.loads((out / "launch.json").read_bytes())
    options = launch["options"]
    mapped_path = Path(completed["mapped_requests_artifact"]["path"])
    if stats.identity(mapped_path) != completed["mapped_requests_artifact"]:
        raise ValueError("Original mapped request source changed")
    mapped = json.loads(mapped_path.read_bytes())["requests"]
    case_path = Path(completed["case_pack"]["artifact"]["path"])
    if stats.identity(case_path) != completed["case_pack"]["artifact"]:
        raise ValueError("Original random-position case asset changed")
    probe_cases = json.loads(case_path.read_bytes())["rank_cases"]
    probes = {(c["traj_uid"], c["source_step"]): c for cases in probe_cases for c in cases}
    credit_path = Path(completed["inputs"]["credits"]["path"])
    actor_path = Path(completed["inputs"]["actor_minibatch"]["path"])
    for path, key in ((credit_path, "credits"), (actor_path, "actor_minibatch")):
        if stats.identity(path) != completed["inputs"][key]:
            raise ValueError("Original native artifact changed")
    credits, actor = (DataProto.load_from_disk(str(path)) for path in (credit_path, actor_path))
    tokenizer = AutoTokenizer.from_pretrained(options["actor_rollout_ref.model.path"], local_files_only=True)
    sampling = {"temperature": options["actor_rollout_ref.rollout.temperature"],
                "max_tokens": options["data.max_response_length"]}
    readout = EventRatioReadout(None, tokenizer, task="TextCraft", max_steps=options["env.max_steps"],
        max_length=32768, minibatch_size=4, packed_answer_targets=None, invalid_action_penalty_coef=0., sampling=sampling)
    indices = torch.tensor(sorted({int(source) for spans in actor.non_tensor_batch["dt_response_slices"]
                                   for source, _, length in spans if length > 0}), dtype=torch.long)
    capture = mapping_owner.CaptureDT()
    try:
        dt_training_batch.compute_training_credit(credits, capture,
            eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id), source_indices=indices)
    except mapping_owner.CapturedNativeRPC:
        pass
    if capture.requests is None:
        raise ValueError("Original CPU preparation did not reach its existing RPC boundary")
    # Use the original producer's actual clean-owner import path, not another
    # runner/target implementation. Only target packing is called on CPU.
    prepared = json.loads((out / "prepared-diagnostic.json").read_bytes())
    runner_path = Path(next(p for p in prepared["sources"] if p.endswith("qwen35_dense_finite_runner.py")))
    sys.path.insert(0, str(runner_path.parent))
    from qwen35_answer_finite import PackedAnswerTargets

    original_groups = defaultdict(list)
    for item in mapped:
        original_groups[(item["rank"], item["trace"]["owner_batch_index"])].append(item)
    iterations = sorted({item["trace"]["owner_batch_index"] for item in mapped if item["env_step"] == 0})
    rank_payloads = [[], []]
    owner_requests = []
    for rank, partition in enumerate(capture.requests.chunk(2)):
        rows = to_list_of_dict(partition)
        report = {"nonzero_reward_events": 0, "policy_tokens": 0, "actual_row_lengths": []}
        _, requests = readout._prepare_episode(rows, report, partition.batch["dt_complete_return"].tolist())
        # The same stable context sort performed by original episodes line256.
        # Original logged owner_batch_index determines selection; no new groups.
        requests.sort(key=lambda request: request["context_tokens"])
        owner_requests.append(requests)
        ordered_mapped = sorted((item for item in mapped if item["rank"] == rank), key=lambda item: item["request_index"])
        if len(requests) != len(ordered_mapped):
            raise ValueError("Original request count changed")
        for request, item in zip(requests, ordered_mapped):
            selected = torch.cat(tuple(request[name] for name in ("prompt", "actions", "query", "target")))
            if selected.tolist() != item["selected_input_ids"]:
                raise ValueError("Original owner request and previously mapped literal IDs differ")
        for iteration in iterations:
            slots = sorted(original_groups[(rank, iteration)], key=lambda item: item["request_index"])
            requests = [owner_requests[rank][item["request_index"]] for item in slots]
            # Representation-only right extension uses PyTorch's existing pad
            # API with the exact native EOS value and original row ordering.
            selected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["selected_input_ids"], dtype=torch.long) for item in slots],
                batch_first=True, padding_value=tokenizer.eos_token_id)
            reference = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["reference_input_ids"], dtype=torch.long) for item in slots],
                batch_first=True, padding_value=tokenizer.eos_token_id)
            if selected.shape != (4, slots[0]["trace"]["compute_tokens"]):
                raise ValueError("Existing EOS extension does not match original B4 geometry")
            observer = ObservePack()
            try:
                trace_token_attribution(observer, reference, selected, [request["case"] for request in requests],
                    [[0] for _ in requests], packed_answer_targets=PackedAnswerTargets,
                    outcome_token_ids=readout.alphabet.label_ids(tokenizer))
            except CapturedOwnerAttribute:
                pass
            if not hasattr(observer, "pair"):
                raise ValueError("Original packed helper did not reach its owner call")
            positions = torch.arange(selected.shape[1])
            starts = torch.where(reference != selected, positions, selected.shape[1]).amin(-1).tolist()
            # This is a source-derived metadata calculation of c9 lines192--204,
            # not an observed prefix boundary or a new cache implementation.
            local_cut = min(min(starts), int(observer.selection.positions.min())) // 64 * 64
            sample_metadata = []
            for request, slot in zip(requests, slots):
                key = (slot["traj_uid"], slot["env_step"])
                is_first = slot["env_step"] == 0
                probe = probes[key] if is_first else None
                sample_metadata.append({"traj_uid": key[0], "source_step": key[1], "original_request_index": slot["request_index"],
                    "source_start": request["start"], "source_end": request["end"],
                    "context_tokens": request["context_tokens"], "compute_tokens": selected.shape[1],
                    "query_tokens": request["query_tokens"], "observed_return": request["observed_return"],
                    "native_target_case": {"target_ids": request["case"]["target_ids"].tolist(),
                                           "prompt_length": request["case"]["prompt_length"]},
                    "probe_source_positions": probe["probe_source_positions"] if probe else None,
                    "probe_input_positions": probe["selected_input_positions"] if probe else None,
                    "saved_probe_credit": probe["saved_probe_credit"] if probe else None,
                    "probe_scope": "Original fixed random positions; two single-EOS variants" if is_first else "Non-first slot unchanged in all variants to preserve original layout",
                    "original_trace": slot["trace"]})
            rank_payloads[rank].append({"original_owner_batch_index": iteration, "rank": rank,
                "selected_input_ids": selected.tolist(), "reference_input_ids": reference.tolist(),
                "selected_input_ids_sha256": tensor_hash(selected), "reference_input_ids_sha256": tensor_hash(reference),
                "array_hash_encoding": "Contiguous native int64 little-endian tensor bytes",
                "paired_owner_abi": {"shape": list(observer.pair.shape), "sha256": tensor_hash(observer.pair),
                    "attention_mask_shape": list(observer.mask.shape), "attention_mask_sha256": tensor_hash(observer.mask),
                    "mask_all_ones": bool((observer.mask == 1).all()),
                    "endpoint_order": "reference,factual per sample; exact existing trace_token_attribution helper",
                    "selection_positions": observer.selection.positions.tolist(),
                    "selection_samples": observer.selection.samples.tolist(), "selection_labels": observer.selection.labels.tolist(),
                    "selection_paired_positions": observer.selection.paired_positions.tolist(),
                    "original_owner_kwargs": observer.kwargs},
                "coefficient_starts_from_literal_original_ID_difference": starts,
                "prefix_start_local_pre_sync_derived": local_cut,
                "samples": sample_metadata})
    for left, right in zip(*rank_payloads):
        if left["original_owner_batch_index"] != right["original_owner_batch_index"]:
            raise ValueError("Original cross-rank iteration indices differ")
        synchronized = min(left["prefix_start_local_pre_sync_derived"], right["prefix_start_local_pre_sync_derived"])
        left["prefix_start_original_cross_rank_MIN_derived"] = synchronized
        right["prefix_start_original_cross_rank_MIN_derived"] = synchronized
    config_path = Path(os.environ["DT_ENVIRONMENT_JSON"])
    env = json.loads(config_path.read_bytes())["qwen35"]
    flag_keys = ["dt_native_fla_fp16", "dt_answer_compiled", "dt_offload_replay_mixer",
        "dt_gdn_head_batch_size", "dt_compile_gdn_scalar_rules", "dt_pin_replay_host",
        "dt_gdn_gpu_capture_names", "dt_fa_coefficient_suffix", "dt_gdn_coefficient_suffix",
        "dt_compact_gdn_captures", "dt_pin_root_host", "dt_reuse_native_prefix", "dt_dynamic_shapes"]
    first_slots = [sample for rank in rank_payloads for group in rank for sample in group["samples"] if sample["source_step"] == 0]
    result = {"scope": __doc__, "sources": {"script": stats.identity(__file__), "mapped_requests": stats.identity(mapped_path),
        "first_response_probe_cases": stats.identity(case_path), "original_credits": stats.identity(credit_path),
        "original_actor_carrier": stats.identity(actor_path), "environment_config_current_same_parent": stats.identity(config_path),
        "runner": stats.identity(runner_path), "original_readout_prepare_episode": stats.source_identity(readout._prepare_episode),
        "original_RPC_preparation": stats.source_identity(dt_training_batch.compute_training_credit),
        "original_target_packing": stats.source_identity(PackedAnswerTargets),
        "original_interleaved_owner_call": stats.source_identity(trace_token_attribution),
        "original_prefix_sync_callback": stats.source_identity(_Qwen35CausalOwnerView.synchronize_prefix_start),
        "torch_right_pad_representation": stats.source_identity(torch.nn.utils.rnn.pad_sequence),
        "original_native_log": completed["inputs"]["diagnostic_log"]},
        "coverage": {"original_B4_calls_per_rank": [24, 24], "selected_original_iteration_indices": iterations,
            "selected_B4_calls_per_rank": [len(rank_payloads[0]), len(rank_payloads[1])],
            "original_slots_preserved": sum(len(group["samples"]) for rank in rank_payloads for group in rank),
            "first_response_slots": len(first_slots), "first_response_unique_UIDs": len({x["traj_uid"] for x in first_slots}),
            "unique_fixed_random_token_probes": len({(x["traj_uid"], pos) for x in first_slots for pos in x["probe_input_positions"]})},
        "sampling": sampling, "max_steps": options["env.max_steps"], "outcome_token_ids": readout.alphabet.label_ids(tokenizer),
        "eos_token_id": tokenizer.eos_token_id, "rank_groups": rank_payloads,
        "native_flag_scope": "Read-only current c9 environment referenced by the same original parent/stager; trace did not store flags or actual prefix cuts",
        "native_flags_current_same_parent": {key: env.get(key) for key in flag_keys},
        "prefix_scope": "Metadata derived from literal firstchanged positions and original PackedAnswerTargets positions under c9 runner192--204 and original producer full-value MIN164--176. Not historical on-device observation. c9 uses a fresh per-B4 native shared prefix/fork; no external flat prefix bank/provider placement",
        "proposed_phase_counts_only": {"variants": ["original_full_response_EOS_baseline", "single_probe0_EOS", "single_probe1_EOS"],
            "maximum_attribute_calls_per_rank": len(iterations)*3, "maximum_native_prefix_forwards_per_rank": len(iterations)*3,
            "maximum_native_root_forwards_per_rank": len(iterations)*3,
            "finite_decoder_or_replay_calls": 0,
            "scope": "Root agent proposed diagnostic stops at original answer callable after categorical native endpoints; none of these model calls are executed by preparation"},
        "limits": ["No alternate logical requests or groups; all non-first original slots are retained unchanged.",
            "Root must verify original fullEOS baseline endpoints before interpreting single deletion comparisons.",
            "Original c9 attribute calls model.forward_root for its cached prefix, not runner.forward_prefix. Its existing model.synchronize_prefix_start callback is the actual cut boundary.",
            "No GPU, model, finite, optimizer, new tolerance or production changes occurred."],
        "runtime": {"elapsed_seconds": time.time()-start, "rss_bytes": psutil.Process().memory_info().rss,
            "pss_bytes": psutil.Process().memory_full_info().pss,
            "cuda_initialized": torch.cuda.is_initialized(), "distributed_initialized": torch.distributed.is_initialized()}}
    path = out / "native64-first-response-matched-layout-pack.json"
    path.write_text(json.dumps(stats.json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": stats.identity(path), "coverage": result["coverage"], "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
