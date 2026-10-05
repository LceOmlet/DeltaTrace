"""CPU dispatch contracts, not GPU/model/tolerance acceptance.

Execute original LOOP submission method plus candidate IPC functions. The
unavailable Torch/Ray/Qwen clients are dependency doubles; no scheduler or
completion-client implementation is substituted in production.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event
import time
from types import ModuleType, SimpleNamespace as NS
import uuid

import pytest


REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "research/temporary/rl_upstream_alignment_20260929"
BASELINE = AUDIT / "appworld-eval-client-routing-20261005/baseline"
CANDIDATE = AUDIT / "appworld-eval-client-routing-20261005/candidate-v2"
LOOP = AUDIT / "recipe-sources/ml-loop/phi_agents/rl/vllm_rollout_worker.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(content):
    return {n.name: n for n in ast.walk(ast.parse(content)) if isinstance(n, ast.FunctionDef)}


def tree_dump(node):
    return ast.dump(node, include_attributes=False)


def module(monkeypatch, name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, result)
    return result


def namespace(value):
    if isinstance(value, dict):
        return NS(**{k: namespace(v) for k, v in value.items()})
    return value


def container(value):
    if isinstance(value, NS):
        return {k: container(v) for k, v in vars(value).items()}
    return value


def submission_method(patched=False):
    source = LOOP.read_text(encoding="utf-8")
    if patched:
        sys.path.insert(0, str(REPO / "experiments/rl"))
        try:
            patch = load(REPO / "experiments/rl/patch_loop_external_completion.py", "external_completion_patch")
            source = patch.patch(source)
        finally:
            sys.path.pop(0)
    fn = functions(source)["_submit_scenarios_and_get_rollouts"]
    env = dict(Queue=Queue, Empty=Empty, ThreadPoolExecutor=ThreadPoolExecutor,
        as_completed=as_completed, logger=NS(info=lambda *a: None))
    exec(compile("from __future__ import annotations\n" + ast.unparse(fn), str(LOOP), "exec"), env)
    return env[fn.name], env, fn


def test_frozen_sources_patch_only_three_files_and_no_owner_scheduler_copy(tmp_path):
    prep = load(AUDIT / "prepare_appworld_eval_client_routing_20261005.py", "eval_routing_prep")
    originals = {name: (BASELINE / name).read_bytes() for name in prep.EXPECTED}
    patched = prep.patched_sources(originals)
    assert patched == {name: (CANDIDATE / name).read_bytes() for name in prep.EXPECTED}
    assert all(hashlib.sha256(originals[name]).hexdigest() == sha for name, sha in prep.EXPECTED.items())
    dest = tmp_path / "candidate"
    assert prep.prepare(BASELINE, dest)["state"] == "prepared-only"
    with pytest.raises(FileExistsError):
        prep.prepare(BASELINE, dest)
    changed = dict(originals)
    changed["loop_owner_worker.py"] += b"\n"
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        prep.patched_sources(changed)
    assert "n_tasks" not in patched["loop_owner_worker.py"].decode()
    for name, unchanged in {
        "loop_owner_worker.py": ["recorded_runner", "event", "queued_event", "start", "close"],
        "loop_async_transport.py": ["response_carrier"],
        "loop_owner_rollout.py": ["close", "to_batch"],
    }.items():
        before, after = functions(originals[name]), functions(patched[name])
        for function in unchanged:
            assert tree_dump(before[function]) == tree_dump(after[function])
    before = functions(LOOP.read_text())["_submit_scenarios_and_get_rollouts"]
    _, _, after = submission_method(patched=True)
    start = lambda fn: next(i for i, n in enumerate(fn.body)
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "futures")
    assert [tree_dump(n) for n in before.body[start(before):]] == [tree_dump(n) for n in after.body[start(after):]]
    candidate = functions(patched["loop_owner_worker.py"])
    assert candidate["run_owner"].args.defaults[-1].value == 1
    pool = next(n for n in ast.parse(patched["loop_owner_worker.py"]).body
        if isinstance(n, ast.ClassDef) and n.name == "OwnerProcesses")
    init = next(n for n in pool.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert init.args.defaults[-1].value == 1
    calls = [n for n in ast.walk(candidate["run_owner"]) if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "VLLMQwen3"]
    assert len(calls) == 2 and tree_dump(calls[0]) == tree_dump(calls[1])


def execute_owner(monkeypatch, path, evaluation, inference_clients=None, scenarios=3, rollouts=2):
    """Run candidate run_owner; original submit chooses actual callback instance."""
    owner = load(path, "cpu_owner")
    built, emitted, kwargs_seen = [], [], []
    replies, commands, cancellation = Queue(), Queue(), Event()
    commands.put("collect"); commands.put("stop")
    method, env, _ = submission_method(patched=True)

    class Client:
        def __init__(self, **kwargs):
            self.index = len(built)
            self._vllm = NS(_is_max_tokens_stopped=lambda x: x["finish_reason"] == "length")
            built.append(self); kwargs_seen.append(kwargs)

    class Sampler:
        def __init__(self, factory, num_threads):
            self.items = iter(["eval-task-alpha", "eval-task-beta", "eval-task-gamma"]
                if scenarios == 3 else [f"official-eval-slot-{i}" for i in range(scenarios)])
        def __next__(self): return next(self.items)
        def ensure_scenarios_requested(self, number): assert number == scenarios
        def stop(self): pass

    class Runner:
        def run(self, scenario, llm):
            owner._recording.requests = []
            result = llm._vllm._completion_transport(dict(prompt=[17, 23], model="policy", stream=False,
                temperature=0, max_tokens=1500))
            assert result == ("returned", [31, 37], [-.2, -.3], False, False)
            return NS(scenario=scenario, selected=llm.index)

    class Worker:
        _submit_scenarios_and_get_rollouts = method
        def __init__(self, **kwargs):
            self._external_llm_factory = kwargs["external_llm_factory"]
            self._cancellation_event = kwargs["external_cancellation_event"]
            self._scenario_sampler = kwargs["scenario_sampler"]
            self._rollouts_per_scenario = kwargs["rollouts_per_scenario"]
            self._local_rank = kwargs["local_rank"]
            self._scenario_runners = [Runner()]
            self.result = None
        def request_rollout_generation(self, count, adapter):
            self.result = list(self._submit_scenarios_and_get_rollouts(count))
        def get_rollouts(self, count, **kwargs):
            emitted.append(("get_options", kwargs))
            return self.result
        def stop(self): pass

    class Output:
        def put(self, event):
            emitted.append(event)
            if event[0] == "completion":
                replies.put((event[2], NS(text="returned", token_ids=[31, 37],
                    logprobs=[-.2, -.3], finish_reason="stop")))

    dist = module(monkeypatch, "torch.distributed", init_process_group=lambda *a, **k: None,
        barrier=lambda: None, is_initialized=lambda: False, destroy_process_group=lambda: None)
    module(monkeypatch, "torch", set_num_threads=lambda n: None, device=lambda d: d, distributed=dist)
    module(monkeypatch, "hydra.utils", instantiate=lambda x: x)
    module(monkeypatch, "omegaconf", OmegaConf=NS(create=namespace, to_container=lambda x, **k: container(x)))
    module(monkeypatch, "phi_agents.rl.parallel_scenario_sampler", ParallelScenarioSampler=Sampler)
    module(monkeypatch, "phi_agents.rl.vllm_rollout_worker", VLLMRolloutWorker=Worker)
    module(monkeypatch, "phi_agents.rl.llm.qwen_3", VLLMQwen3=Client)
    config = dict(rl=dict(eval=dict(scenario_sampler={}), scenario_sampler={}, scenario_runner={},
        params=dict(rollouts_per_scenario=rollouts, scenarios_per_iteration=scenarios),
        rollouts_fraction=.9, rollouts_per_scenario_fraction=.75, num_scenario_runners=1),
        llm=dict(vllm_class={"_target_": "phi_agents.rl.llm.qwen_3.VLLMQwen3", "test_client_flag": 7},
            temperature=0, vllm_server=dict(max_model_len=32768)))
    try:
        args = (0, 1, "unused-cpu", config, "recorded-model-path", 0, evaluation,
            commands, replies, Output(), cancellation)
        owner.run_owner(*args, **({"inference_clients": inference_clients}
            if path.parent == CANDIDATE and inference_clients is not None else {}))
    finally:
        replies.put(None)
    assert not any(event[0] == "error" for event in emitted), emitted
    completions = [event for event in emitted if event[0] == "completion"]
    selected = [row[2].selected for event in emitted if event[0] == "result" for row in event[2]]
    return built, kwargs_seen, completions, selected, emitted


def test_original_owner_selection_and_candidate_eval_clients(monkeypatch):
    built, kwargs, completions, selected, events = execute_owner(monkeypatch,
        CANDIDATE / "loop_owner_worker.py", True, inference_clients=2, scenarios=57)
    assert len(built) == 2 and kwargs[0] == kwargs[1]
    assert len(completions) == 114
    assert [event[4] for event in completions] == selected
    # Expected sequence is obtained by executing the unpatched original owner,
    # not by implementing its selection expression in the test/adapter.
    original, env, _ = submission_method()
    env["hydra"] = NS(utils=NS(instantiate=lambda _, **kw: built[kw["port"]]))
    class Sampler:
        def __init__(self): self.items = iter(range(57))
        def __next__(self): return next(self.items)
        def ensure_scenarios_requested(self, count): assert count == 57
    fixture = NS(_hosts=["server", "server"], _vllm_server_ports=[0, 1],
        _llm_cfg=NS(vllm_class={}, temperature=0), _local_base_model_path=Path("recorded-model-path"),
        _loaded_adapter_model_id=None, _vllm_server_cfg=NS(max_model_len=32768), _cancellation_event=Event(),
        _scenario_sampler=Sampler(), _rollouts_per_scenario=2, _local_rank=0,
        _scenario_runners=[NS(run=lambda scenario, llm: llm.index)])
    original_selected = [value for _, _, value in original(fixture, 57)]
    assert selected == original_selected
    assert selected.count(0) == selected.count(1) == 57
    assert [e[1] for e in events if e[0] == "get_options"] == [dict(world_size=1, device="cpu")]


def test_training_default_same_as_baseline_even_with_multiple_available_servers(monkeypatch):
    before = execute_owner(monkeypatch, BASELINE / "loop_owner_worker.py", False)
    after = execute_owner(monkeypatch, CANDIDATE / "loop_owner_worker.py", False, inference_clients=2)
    assert len(before[0]) == len(after[0]) == 1
    assert before[1] == after[1] and before[3] == after[3]
    normalize = lambda events: [(e[0], e[1], e[3:]) if e[0] == "completion" else e
        for e in events if e[0] in ("completion", "get_options")]
    assert normalize(before[4]) == normalize(after[4])
    assert all(len(e) == 4 for e in after[2])


def test_default_eval_client_preserves_four_field_completion(monkeypatch):
    before = execute_owner(monkeypatch, BASELINE / "loop_owner_worker.py", True)
    after = execute_owner(monkeypatch, CANDIDATE / "loop_owner_worker.py", True)
    assert len(before[0]) == len(after[0]) == 1
    assert before[1] == after[1] and before[3] == after[3]
    assert all(len(event) == 4 for event in after[2])
    assert [event[3] for event in before[2]] == [event[3] for event in after[2]]


def test_default_eval_events_reach_existing_sync_collector(monkeypatch):
    """Execute the existing synchronous collector, with transport-only doubles."""
    import numpy as np

    _, _, completions, _, _ = execute_owner(monkeypatch,
        CANDIDATE / "loop_owner_worker.py", True)
    queue, replies, cancellation = Queue(), Queue(), Event()
    for event in completions:
        queue.put(event)
    queue.put(("result", 0, ([], [])))
    pool = NS(processes=[object()], output=queue, replies=[replies],
        cancellations=[cancellation], start=lambda: None,
        event=lambda: queue.get(timeout=2), queued_event=queue.get_nowait)
    remote_calls = []

    class Batch:
        def __init__(self, count, non_tensors=None, meta_info=None):
            self.count = count
            self.non_tensor_batch = dict(non_tensors or {})
            self.meta_info = dict(meta_info or {})
        def __len__(self): return self.count
        def chunk(self, count):
            assert count == 1
            return [self]
        def select_idxs(self, indices):
            assert list(indices) == list(range(self.count)) or len(indices) == 1
            return self

    class Ref:
        def __init__(self, batch): self.batch = batch
        def future(self): return self
        def add_done_callback(self, callback): callback(self)

    class Engines:
        def __init__(self, workers): self.pending = []
        def has_free(self): return not self.pending
        def has_next(self): return bool(self.pending)
        def submit(self, callback, batch): self.pending.append(callback("native-worker", batch).batch)
        def get_next_unordered(self, timeout): return self.pending.pop(0)

    def execute_remote(worker, method, batch):
        remote_calls.append((worker, method, batch))
        return Ref(batch)

    module(monkeypatch, "loop_owner_worker", OwnerProcesses=lambda *args: None)
    module(monkeypatch, "owner_environment_transport", policy_reply=lambda batch, index:
        NS(token_ids=[31, 37], text="returned", logprobs=[-.2, -.3], finish_reason="stop"))
    module(monkeypatch, "verl.protocol", pad_dataproto_to_divisor=lambda batch, divisor: (batch, 0))
    module(monkeypatch, "ray.util.actor_pool", ActorPool=Engines)
    torch = NS(long="long", zeros=lambda count, width, **kwargs: [None] * count,
        tensor=lambda indices, **kwargs: indices)
    data_proto = NS(from_dict=lambda tensors, non_tensors, meta_info:
        Batch(len(tensors["input_ids"]), non_tensors, meta_info))
    method = functions((CANDIDATE / "loop_owner_rollout.py").read_bytes())["collect_native_trajectories"]
    env = dict(Empty=Empty, np=np, torch=torch, DataProto=data_proto, time=time, uuid=uuid)
    exec(compile("from __future__ import annotations\n" + ast.unparse(method), "sync-owner-entry", "exec"), env)
    owner = NS(processes=pool, tokenizer=NS(eos_token_id=2, pad_token_id=0),
        to_batch=lambda rows, records: (rows, records))
    collector = NS(preprocess_batch=lambda carrier, raw: carrier)
    group = NS(workers=["native-worker"], world_size=1, generate_sequences=object(),
        _execute_remote_single_worker=execute_remote)
    rows, records = env[method.name](owner, collector, NS(meta_info={}), group, False)
    assert rows == [] and len(remote_calls) == 1
    assert set(records) == {event[2] for event in completions}
    assert [key for key, _ in replies.queue] == [event[2] for event in completions]


def transport(monkeypatch, path, events, cancel_after_result=False):
    """Execute actual collect function with synchronous CPU RPC/carrier doubles."""
    worker = load(path, "cpu_transport")
    queue, reply_queue, cancelled = Queue(), Queue(), Event()
    for event in events: queue.put(event)
    calls, aborts, preprocess = [], [], []
    class Future:
        def __init__(self, key): self.key = key
        def add_done_callback(self, callback): callback(self)
    class Ref:
        def __init__(self, index, key): self.index, self.key = index, key
        def future(self): return Future(self.key)
    def server(index):
        def generate(ids, options, key):
            calls.append((index, ids, options, key)); return Ref(index, key)
        return NS(generate_tokens=NS(remote=generate), abort_request=NS(remote=lambda key: aborts.append((index, key))))
    def event():
        value = queue.get(timeout=2)
        if cancel_after_result and value[0] == "result": cancelled.set()
        return value
    pool = NS(processes=[object()], start=lambda: None, event=event, output=queue,
        cancellations=[cancelled, Event()], replies=[reply_queue, reply_queue])
    servers = [server(0), server(1)]
    owner = NS(processes=pool, tokenizer=NS(pad_token_id=0, eos_token_id=2),
        config=NS(data=NS(max_response_length=1500)), to_batch=lambda rows, records: (rows, records))
    def prepare(carrier, raw):
        preprocess.append(raw); return NS(raw=raw, carrier=carrier)
    collector = NS(async_rollout_manager=NS(async_llm_servers=servers), preprocess_batch=prepare)
    module(monkeypatch, "torch", zeros=lambda *a, **k: [[0]], long="long")
    module(monkeypatch, "ray", get=lambda ref: NS(index=ref.index, key=ref.key))
    module(monkeypatch, "verl", DataProto=NS(from_dict=lambda **kw: NS(**kw)))
    module(monkeypatch, "owner_environment_transport", policy_reply=lambda row, index:
        NS(token_ids=[31, 37], text="returned", logprobs=[-.2, -.3], finish_reason="stop"))
    worker.response_carrier = lambda prompt, output, *args: NS(prompt=prompt, output=output)
    result = worker.collect(owner, collector, NS(meta_info={"sampling": "original"}))
    return calls, aborts, preprocess, list(reply_queue.queue), result


def test_actual_collect_routes_selected_eval_server_and_original_reply_rank(monkeypatch):
    raw = dict(prompt=[17, 23], model="policy", stream=False, temperature=0, max_tokens=1500)
    events = [("completion", 0, "eval-0", raw, 0), ("completion", 0, "eval-1", raw, 1),
        ("result", 0, (["task"], [["rollout"]]))]
    calls, aborts, preprocess, replies, result = transport(monkeypatch, CANDIDATE / "loop_async_transport.py", events)
    assert [(index, key) for index, _, _, key in calls] == [(0, "eval-0"), (1, "eval-1")]
    assert not aborts and [key for key, reply in replies] == ["eval-0", "eval-1"]
    assert all(value["raw_prompt_ids"] == [[17, 23]] for value in preprocess)
    assert set(result[1]) == {"eval-0", "eval-1"}


def test_actual_collect_cancel_uses_reply_rank_and_aborts_selected_server(monkeypatch):
    raw = dict(prompt=[17, 23], model="policy", stream=False)
    events = [("completion", 0, "cancel-server-1", raw, 1), ("result", 0, ([], []))]
    calls, aborts, _, replies, result = transport(monkeypatch, CANDIDATE / "loop_async_transport.py",
        events, cancel_after_result=True)
    assert calls[0][0] == 1 and aborts == [(1, "cancel-server-1")]
    assert not replies and result[1] == {}


def test_actual_collect_four_field_training_matches_baseline(monkeypatch):
    raw = dict(prompt=[17, 23], model="policy", stream=False, temperature=.7)
    events = [("completion", 0, "train-0", raw), ("completion", 1, "train-1", raw),
        ("result", 0, ([], []))]
    before = transport(monkeypatch, BASELINE / "loop_async_transport.py", events)
    after = transport(monkeypatch, CANDIDATE / "loop_async_transport.py", events)
    assert before[:4] == after[:4]
    assert set(before[4][1]) == set(after[4][1])


@pytest.mark.parametrize("evaluation,expected_clients,expected_world", [(True, 2, 1), (False, 1, 2)])
def test_actual_rollout_async_entry_passes_server_count_only_for_eval(monkeypatch, evaluation,
        expected_clients, expected_world):
    called = []
    module(monkeypatch, "loop_owner_worker", OwnerProcesses=lambda *args: called.append(args) or NS())
    module(monkeypatch, "loop_async_transport", collect=lambda owner, collector, batch: "original-async-collect")
    module(monkeypatch, "owner_environment_transport", policy_reply=lambda *a: None)
    module(monkeypatch, "verl.protocol", pad_dataproto_to_divisor=lambda *a: None)
    module(monkeypatch, "ray.util.actor_pool", ActorPool=object)
    method = functions((CANDIDATE / "loop_owner_rollout.py").read_bytes())["collect_native_trajectories"]
    env = {"OmegaConf": NS(to_container=lambda value, **kw: value)}
    exec(compile("from __future__ import annotations\n" + ast.unparse(method), "native-owner-entry", "exec"), env)
    owner = NS(processes=None, evaluation=evaluation, native={"official_config": True}, reserve=0,
        config=NS(trainer=NS(n_gpus_per_node=2, nnodes=1), actor_rollout_ref=NS(model=NS(path="model"))))
    collector = NS(async_rollout_manager=NS(async_llm_servers=["native0", "native1"]))
    assert env[method.name](owner, collector, object(), object(), not evaluation) == "original-async-collect"
    assert called == [({"official_config": True}, "model", 0, evaluation, expected_world, expected_clients)]
