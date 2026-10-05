"""Read-only CPU alignment of existing native64 DT reports and owner artifacts.

The original credit preparation/distribution runs only up to its RPC boundary.
The RPC captures its original DataProto and raises; it returns no fake credit.
No model, trainer, inference engine, optimizer or distributed group is created.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import psutil
import torch
from transformers import AutoTokenizer
from verl.protocol import DataProto
from agent_system.multi_turn_rollout.utils import to_list_of_dict
import counterfactual
import dt_training_batch
from reward_readout import EventRatioReadout
import analyze_textcraft_native_minibatch as analyzer


class CapturedNativeRPC(Exception):
    """Stop at the existing DT RPC, without fabricating its return value."""


class CaptureDT:
    world_size = 2

    def __init__(self):
        self.requests = None

    def compute_dt_token_advantages(self, data):
        self.requests = data
        raise CapturedNativeRPC()


def resources():
    process = psutil.Process()
    return {"rss_bytes": process.memory_info().rss,
            "pss_bytes": getattr(process.memory_full_info(), "pss", None),
            "available_host_bytes": psutil.virtual_memory().available}


def write(path, value):
    path.write_text(json.dumps(analyzer.json_safe(value), indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return analyzer.identity(path)


def logged_reports(path):
    reports, pid_to_rank = [], {}
    for number, line in enumerate(path.open(encoding="utf-8", errors="replace"), 1):
        pid_match = re.search(r"\(WorkerDict pid=(\d+)\)", line)
        pid = int(pid_match.group(1)) if pid_match else None
        rank_match = re.search(r"native_minibatch_diagnostic rank=(\d+)\b", line)
        if pid is not None and rank_match:
            rank = int(rank_match.group(1))
            if pid in pid_to_rank and pid_to_rank[pid] != rank:
                raise ValueError("Original WorkerDict PID has conflicting logged ranks")
            pid_to_rank[pid] = rank
        marker = "[DeltaTrace readout] "
        if marker in line:
            reports.append({"line": number, "pid": pid,
                            "report": json.loads(line.split(marker, 1)[1])})
    by_rank = {}
    for item in reports:
        if item["pid"] not in pid_to_rank:
            raise ValueError("The original report PID is not bound to a native gradient rank")
        rank = pid_to_rank[item["pid"]]
        if rank in by_rank:
            raise ValueError("More than one original readout report for this rank")
        by_rank[rank] = item
    if set(by_rank) != {0, 1}:
        raise ValueError("The diagnostic log does not contain both original rank reports")
    return by_rank, pid_to_rank


def scalar_summary(values):
    return analyzer.summary(torch.tensor(values, dtype=torch.float64), torch)


def original_source_parts(row):
    """Representation-only extraction used by the original _prepare_episode."""
    response = row["responses"]
    width = response.numel()
    attention = row["attention_mask"].bool()
    positions = attention[-width:].nonzero().flatten().tolist()
    if positions != list(range(len(positions))):
        raise ValueError("Original response is not right padded")
    if not torch.equal(row["input_ids"][-width:], response):
        raise ValueError("Original input and response token IDs differ")
    return row["input_ids"][:-width][attention[:-width]], response[:len(positions)]


def saved_credit(actor_data, actor_row, start, length, g):
    fields = {name: actor_data.batch[name][actor_row, start:start + length]
              for name in ("advantages", "dt_token_advantages", "dt_q_estimates", "dt_v_estimates")}
    if not torch.equal(fields["advantages"], fields["dt_token_advantages"]):
        raise ValueError("Saved actor advantage differs from its original DT field")
    if g == 0:
        return fields, None, {"inverse_scope": "Undefined at G=0; no d is fabricated"}
    # Algebraic inversion of an already-saved A, for diagnosis only. The forward
    # recomposition below uses the existing owner; this creates no new credit.
    d = -torch.log1p(-fields["advantages"].double() / g)
    owner = counterfactual.reward_event_token_credit(
        d.float()[None, None, :], torch.tensor([[g]], dtype=torch.float32),
        torch.ones((1, 1, length), dtype=torch.bool),
        torch.ones((1, length), dtype=torch.bool))
    return fields, d, {
        "inverse_scope": "-log1p(-saved_A/G) in FP64; saved A was FP32, so this does not recover the pre-storage signed vector exactly",
        "d_from_saved_A": analyzer.summary(d, torch),
        "owner_recomposition_max_abs": {
            "A": float((owner.advantages[0] - fields["dt_token_advantages"]).abs().max()),
            "Q": float((owner.q_estimates[0] - fields["dt_q_estimates"]).abs().max()),
            "V": float((owner.v_estimates[0] - fields["dt_v_estimates"]).abs().max()),
        },
    }


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    torch.set_num_threads(1)
    result = {"scope": __doc__, "started_unix": started,
              "runtime": {"pid": os.getpid(), "pid_birth": psutil.Process().create_time(),
                          "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
                          "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                          "resource_before": resources()}, "status": "started"}
    result_path = out / "native-minibatch-readout-mapping.json"
    try:
        prepared = json.loads((out / "prepared-diagnostic.json").read_bytes())
        launch = json.loads((out / "launch.json").read_bytes())
        options = launch["options"]
        entry = Path(next(p for p in prepared["sources"] if p.endswith("/entry/textcraft_owner_rollout.py"))).parent
        source_checks = {p: analyzer.identity(p) for p in prepared["sources"]}
        for p, expected in prepared["sources"].items():
            if source_checks[p]["sha256"] != expected:
                raise ValueError("Prepared source changed: " + p)
        # Match the original diagnostic stager's two-field DT_SAMPLING_JSON,
        # which is also preserved literally in the logged query. Never inherit
        # AppWorld's environment or add other author sampler fields to the query.
        config_path = entry / "owner_environment_configs.json"
        official_config = json.loads(config_path.read_bytes())["TextCraft"]
        rollout = official_config["train"]["actor_rollout_ref"]["rollout"]
        sampling = {"temperature": options["actor_rollout_ref.rollout.temperature"],
                    "max_tokens": options["data.max_response_length"]}
        if sampling["temperature"] != rollout["temperature"]:
            raise ValueError("TextCraft launch and original sampling temperature differ")
        if sampling["max_tokens"] != rollout["max_tokens"]:
            raise ValueError("TextCraft launch and original sampling max_tokens differ")
        max_steps = options["env.max_steps"]
        model_path = options["actor_rollout_ref.model.path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        readout = EventRatioReadout(None, tokenizer, task="TextCraft", max_steps=max_steps,
                                   max_length=32768, minibatch_size=4, packed_answer_targets=None,
                                   invalid_action_penalty_coef=0.0, sampling=sampling)
        snapshots = list((out / "native-collect").glob("*/credit_responses.pkl"))
        if len(snapshots) != 1:
            raise ValueError("Expected the one existing native64 credit response snapshot")
        credit_path = snapshots[0]
        actor_path = out / "native-optimizer-minibatch.pkl"
        credits = DataProto.load_from_disk(str(credit_path))
        actor_data = DataProto.load_from_disk(str(actor_path))
        if any(value.device.type != "cpu" for data in (credits, actor_data) for value in data.batch.values()):
            raise ValueError("Original DataProto restore is not CPU-only")
        if len(actor_data) != 64 or len(set(actor_data.non_tensor_batch["traj_uid"].tolist())) != 64:
            raise ValueError("Saved actor carrier is not the original64 unique trajectories")
        result.update(inputs={"credits": analyzer.identity(credit_path), "actor_minibatch": analyzer.identity(actor_path),
                              "diagnostic_log": analyzer.identity(out / "diagnostic.log"),
                              "launch": analyzer.identity(out / "launch.json"), "prepared": analyzer.identity(out / "prepared-diagnostic.json")},
                      sources={"script": analyzer.identity(__file__), "summary": analyzer.source_identity(analyzer.summary),
                               "protocol_load": analyzer.source_identity(DataProto.load_from_disk),
                               "protocol_chunk": analyzer.source_identity(DataProto.chunk),
                               "to_list_of_dict": analyzer.source_identity(to_list_of_dict),
                               "prepare_training_credit": analyzer.source_identity(dt_training_batch.prepare_training_credit),
                               "compute_training_credit": analyzer.source_identity(dt_training_batch.compute_training_credit),
                               "readout_prepare_episode": analyzer.source_identity(readout._prepare_episode),
                               "alphabet_query_ids": analyzer.source_identity(readout.alphabet.query_ids),
                               "reward_composition": analyzer.source_identity(counterfactual.reward_event_token_credit),
                               "prepared_files": source_checks, "official_environment_config": analyzer.identity(config_path)},
                      readout_config={"sampling": sampling, "max_steps": max_steps, "max_length":32768,
                                      "model_tokenizer_path": model_path, "tokenizer_local_files_only": True,
                                      "outcome_token_ids": readout.alphabet.label_ids(tokenizer)})
        source_rows = to_list_of_dict(credits)
        source, inverse = dt_training_batch.prepare_training_credit(credits)
        complete_returns = source.batch["dt_complete_return"]
        slices_by_source = defaultdict(list)
        for actor_row, spans in enumerate(actor_data.non_tensor_batch["dt_response_slices"]):
            uid = str(actor_data.non_tensor_batch["traj_uid"][actor_row])
            for raw_source, start, length in spans:
                raw_source, start, length = int(raw_source), int(start), int(length)
                if uid != str(source_rows[raw_source]["traj_uid"]):
                    raise ValueError("Original source slice UID and balanced actor UID differ")
                if length:
                    _, actions = original_source_parts(source_rows[raw_source])
                    if not torch.equal(actor_data.batch["responses"][actor_row, start:start + length].long(), actions[:length].long()):
                        raise ValueError("Saved actor slice and original response IDs differ")
                slices_by_source[raw_source].append((actor_row, start, length))
        indices = torch.tensor(sorted({i for i, spans in slices_by_source.items() if any(x[2] > 0 for x in spans)}), dtype=torch.long)
        capture = CaptureDT()
        try:
            dt_training_batch.compute_training_credit(credits, capture,
                eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id), source_indices=indices)
        except CapturedNativeRPC:
            pass
        if capture.requests is None:
            raise ValueError("Original preparation did not reach the DT RPC boundary")
        reports, pid_ranks = logged_reports(out / "diagnostic.log")
        mapped, uid_stats = [], defaultdict(list)
        original_by_identity = {}
        for i, row in enumerate(source_rows):
            original_by_identity.setdefault((str(row["traj_uid"]), int(row["env_step"])), []).append(i)
        for rank, partition in enumerate(capture.requests.chunk(2)):
            rows = to_list_of_dict(partition)
            dummy_report = {"nonzero_reward_events":0, "policy_tokens":0, "actual_row_lengths":[]}
            _, requests = readout._prepare_episode(rows, dummy_report, partition.batch["dt_complete_return"].tolist())
            requests.sort(key=lambda request:request["context_tokens"])
            traces = reports[rank]["report"]["traces"]
            if len(requests) != len(traces):
                raise ValueError(f"rank{rank} original request count {len(requests)} != logged {len(traces)}")
            for request_index, (request, trace) in enumerate(zip(requests, traces)):
                batch_index = request_index // readout.minibatch_size
                width = max(x["context_tokens"] for x in requests[batch_index*4:batch_index*4+4])
                expected = {"source_step":request["source_step"], "context_tokens":request["context_tokens"],
                            "query_tokens":request["query_tokens"], "observed_return":request["observed_return"],
                            "compute_tokens":width, "owner_batch_index":batch_index, "episode_index":0}
                for key, value in expected.items():
                    if trace[key] != value:
                        raise ValueError(f"rank{rank} request{request_index} field{key} differs: {trace[key]} / {value}")
                identity = (request["traj_uid"], request["source_step"])
                raw_indices = original_by_identity[identity]
                retained = []
                for raw_source in raw_indices:
                    for actor_row, start, length in slices_by_source.get(raw_source, []):
                        if not length:continue
                        g = float(complete_returns[int(inverse[raw_source])])
                        fields, d, numeric = saved_credit(actor_data, actor_row, start, length, g)
                        full = length == request["actions"].numel()
                        numeric.update(actor_row=actor_row, actor_response_start=start, retained_tokens=length,
                            full_response_retained=full, actual_G=g,
                            saved_A_summary=analyzer.summary(fields["advantages"],torch),
                            inferred_d_sum_minus_logged_signed_sum=(float(d.sum())-trace["signed_sum"] if full else None),
                            comparison_scope="Only full retained response sums are compared; inverse FP32 A does not recover the original FP64 signed entries exactly")
                        retained.append(numeric)
                factual_ids = torch.cat(tuple(request[name] for name in ("prompt","actions","query","target")))
                reference_ids = factual_ids.clone()
                reference_ids[request["start"]:request["end"]] = tokenizer.eos_token_id
                item = {"rank":rank,"report_log_line":reports[rank]["line"],"worker_pid":reports[rank]["pid"],
                        "request_index":request_index,"traj_uid":identity[0],"env_step":identity[1],
                        "original_credit_row_indices":raw_indices,"source_start":request["start"],"source_end":request["end"],
                        "trace":trace,"field_alignment":expected,"saved_actor_slices":retained,
                        "selected_input_ids":factual_ids.tolist(),"reference_input_ids":reference_ids.tolist(),
                        "dense_eos_extension":width-request["context_tokens"],"target_id":int(request["target"][0]),
                        "factual_probability":math.exp(trace["factual_target_logp"]),
                        "whole_response_EOS_probability":math.exp(trace["reference_target_logp"])}
                mapped.append(item);uid_stats[identity[0]].append(item)
        request_path = out / "readout-mapped-requests.json"
        request_receipt = write(request_path, {"scope":"Exact existing request IDs and original logged root scores, no new model call; transport duplicate slots retained", "requests":mapped})
        # Existing actor ordering and DataProto.chunk own this64-case partition.
        cases_by_actor_row = {}
        random_source = random.Random(20261006)
        for actor_row in range(len(actor_data)):
            uid = str(actor_data.non_tensor_batch["traj_uid"][actor_row])
            eligible = []
            for raw_source, start, length in actor_data.non_tensor_batch["dt_response_slices"][actor_row]:
                raw_source,start,length=int(raw_source),int(start),int(length)
                row = source_rows[raw_source]
                if bool(row["active_masks"]) and length >= 2:
                    eligible.append((int(row["env_step"]), raw_source,start,length))
            if not eligible:
                raise ValueError("Real trajectory has no retained active response with two source tokens: "+uid)
            step,raw_source,start,length=min(eligible)
            row=source_rows[raw_source];prompt,actions=original_source_parts(row)
            g=float(complete_returns[int(inverse[raw_source])])
            query=torch.tensor(readout.alphabet.query_ids(tokenizer,current_step=step,max_steps=max_steps,sampling=sampling),dtype=torch.long)
            label_ids=readout.alphabet.label_ids(tokenizer)
            target=int(label_ids[readout.alphabet.observed_index(g)])
            selected=torch.cat((prompt,actions,query,torch.tensor([target],dtype=torch.long)))
            if selected.numel()>readout.max_length:raise ValueError("Original diagnostic readout input exceeds its unchanged cap")
            positions=random_source.sample(range(length),2)
            fields,d,numeric=saved_credit(actor_data,actor_row,start,length,g)
            metadata={key:row[key] for key in ("traj_uid","env_step","data_source","active_masks","rewards","episode_rewards","episode_lengths") if key in row}
            metadata={key:(value.item() if hasattr(value,'item') else value) for key,value in metadata.items()}
            fingerprint=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            probe_credit=[{"source_position":position,"input_position":prompt.numel()+position,
                "token_id":int(actions[position]),"d":None if d is None else float(d[position]),
                "A":float(fields["advantages"][position]),"Q":float(fields["dt_q_estimates"][position]),
                "V":float(fields["dt_v_estimates"][position]),
                "d_scope":"missing at G0" if d is None else "inferred from saved FP32 A; not original FP64 signed entry"}
                for position in positions]
            cases_by_actor_row[actor_row]={"actor_row":actor_row,"traj_uid":uid,"env_step":step,"source_step":step,"raw_credit_row":raw_source,
                "actual_G":g,"observed_return":g,"observed_class_index":readout.alphabet.observed_index(g),
                "target_id":target,"outcome_token_ids":label_ids,"source_start":prompt.numel(),
                "source_end":prompt.numel()+actions.numel(),"retained_source_tokens":length,
                "query_tokens":query.numel(),"selected_input_ids":selected.tolist(),
                "read_outcomes_input_contract":"Use selected_input_ids[:-1]; its last column is the real target predictor. Four equal-length rows=factual/full response EOS/single1 EOS/single2 EOS; do not append cross-case compute suffix.",
                "selected_source_positions":positions,"probe_source_positions":positions,
                "saved_probe_credit":probe_credit,"selected_input_positions":[prompt.numel()+x for x in positions],
                "selection":"Uniform without replacement from retained source positions using Random(20261006); no DT/extrema selection",
                "saved_source_A":fields["advantages"].tolist(),"saved_source_Q":fields["dt_q_estimates"].tolist(),
                "saved_source_V":fields["dt_v_estimates"].tolist(),"saved_source_d_from_A":None if d is None else d.tolist(),
                "d_missing_scope":None if d is not None else "G0 skipped native DT; value is missing, not zero",
                "original_metadata":metadata,"metadata_sha256":fingerprint,"saved_credit_diagnostic":numeric}
        case_partitions=[]
        for rank,part in enumerate(actor_data.chunk(2)):
            rows=[]
            for uid in part.non_tensor_batch["traj_uid"]:
                matches=[i for i in cases_by_actor_row if cases_by_actor_row[i]["traj_uid"]==str(uid)]
                if len(matches)!=1:raise ValueError("Original actor partition case UID is ambiguous")
                rows.append(cases_by_actor_row[matches[0]])
            case_partitions.append({"rank":rank,"cases":rows})
        case_path=out/'readout-first-response-cases.json'
        case_receipt=write(case_path,{"scope":"Diagnostic input pack only; no credit output, model call or alternative environment score", "sampling":sampling,
            "max_steps":max_steps,"partition_owner":analyzer.source_identity(DataProto.chunk),
            "partition_scope":"Exact existing saved balanced actor DataProto.chunk(2) ordering; no extra sort or scheduler",
            "selection_seed":20261006,
            "selection":{"scope":"Each saved UID's earliest active response with at least two retained source tokens; two distinct positions uniformly drawn without DT/extrema selection", "seed":20261006,
                "position_population":"Original retained generated source-token positions, no observation/query/target positions", "partition":"Original saved actor DataProto.chunk(2)"},
            "cases":64,"rank_cases":[part['cases'] for part in case_partitions],
            "inputs":{"credit_responses":analyzer.identity(credit_path),"minibatch":analyzer.identity(actor_path)}})
        uid_records=[]
        for uid,items in uid_stats.items():
            unique={item['env_step']:item for item in items}
            values=list(unique.values())
            uid_records.append({'traj_uid':uid,'transport_request_slots':len(items),'unique_response_identities':len(unique),
                'factual_probability':scalar_summary([x['factual_probability'] for x in values]),
                'whole_response_EOS_probability':scalar_summary([x['whole_response_EOS_probability'] for x in values]),
                'joint_root':scalar_summary([x['trace']['root_effect'] for x in values])})
        all_cases=list(cases_by_actor_row.values())
        result.update(status='mapped_original_requests_and_prepared_readout_cases',
            captured_native_rpc={'owner':analyzer.source_identity(dt_training_batch.compute_training_credit),
                'rows':len(capture.requests),'stopped_before_model':True,'fake_credit_returned':False,
                'chunk_owner':analyzer.source_identity(DataProto.chunk),'report_pid_to_rank':pid_ranks},
            request_slots=len(mapped),unique_response_identities=len({(x['traj_uid'],x['env_step']) for x in mapped}),
            uid_count_with_existing_nonzero_return_readout=len(uid_stats),per_uid=uid_records,
            aggregate={'factual_probability':scalar_summary([x['factual_probability'] for x in mapped]),
                'whole_response_EOS_probability':scalar_summary([x['whole_response_EOS_probability'] for x in mapped]),
                'joint_root':scalar_summary([x['trace']['root_effect'] for x in mapped])},
            case_pack={'artifact':case_receipt,'case_count':len(all_cases),'per_rank':[len(x['cases']) for x in case_partitions],
                'G_counts':{str(g):sum(c['actual_G']==g for c in all_cases) for g in sorted({c['actual_G'] for c in all_cases})},
                'input_lengths':scalar_summary([len(c['selected_input_ids']) for c in all_cases])},
            mapped_requests_artifact=request_receipt,
            limits=['Existing reports score nonzero-return requests only; zero-return cases in the new pack have no measured root probability.',
                'Whole-response EOS scores are not single-token deletion scores or environmental oracle probabilities.',
                'Native transport padding duplicates remain slots; distinct identities and UID pooling are reported separately.',
                'Saved FP32 A inversion is diagnostic, not an exact reconstruction of the original FP64 finite vector or a numerical tolerance.',
                'No output is a new training credit, normalized advantage, new parameter configuration or production fix.'])
    except Exception as error:
        result.update(status='mapping_failed_no_model_called',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['runtime'].update(resource_after=resources(),cuda_initialized=torch.cuda.is_initialized(),
                                 distributed_initialized=torch.distributed.is_initialized())
        result['completed_unix']=time.time()
        write(result_path,result)
    print(json.dumps({'status':result['status'],'output':analyzer.identity(result_path),
        'request_slots':result['request_slots'],'unique_responses':result['unique_response_identities'],
        'case_pack':result['case_pack'],'resources':result['runtime']},ensure_ascii=False))


if __name__=='__main__':
    main()
