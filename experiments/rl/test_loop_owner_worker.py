"""CPU owner-method, cancellation and exact-token transport contracts."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from threading import Event
import time

import pytest
from omegaconf import OmegaConf


def test_transport_receives_already_queued_item_before_feeder_finishes():
    """Real Queue contract: a counted put need not be available to get_nowait."""
    import multiprocessing as mp
    from queue import Empty
    from threading import Thread, Event
    from loop_owner_worker import OwnerProcesses
    serializing, release = Event(), Event()
    class DelayedItem:
        def __reduce__(self):
            serializing.set()
            assert release.wait(5)
            return tuple, (('completion', 0, 'request', {'prompt': [1, 2, 3]}),)
    queue = mp.get_context('spawn').Queue()
    pool = OwnerProcesses.__new__(OwnerProcesses)
    pool.output, pool.processes = queue, []
    try:
        queue.put(DelayedItem())
        assert serializing.wait(5)
        assert queue.qsize() == 1
        with pytest.raises(Empty):
            queue.get_nowait()
        releaser = Thread(target=lambda: (time.sleep(.05), release.set()))
        releaser.start()
        assert pool.queued_event() == ('completion', 0, 'request', {'prompt': [1, 2, 3]})
        releaser.join()
        with pytest.raises(Empty):
            pool.queued_event()
    finally:
        release.set()
        queue.close()
        queue.join_thread()


def test_original_scheduler_collection_and_cleanup_are_unchanged():
    from phi_agents.rl import vllm_rollout_worker as owner
    old = ast.parse(Path(os.environ['LOOP_ROLLOUT_BASELINE']).read_text())
    new = ast.parse(Path(owner.__file__).read_text())
    methods = lambda tree: {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    a, b = methods(old), methods(new)
    for name in ['get_rollouts', '_worker', 'stop']:
        assert ast.dump(a[name], include_attributes=False) == ast.dump(b[name], include_attributes=False)
    # Only LLM construction is conditional; the original executor/task queue,
    # sampler calls and future completion loop are byte-for-byte AST identical.
    begin = lambda fn: next(i for i,n in enumerate(fn.body)
                            if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '') == 'futures')
    assert [ast.dump(n) for n in a['_submit_scenarios_and_get_rollouts'].body[begin(a['_submit_scenarios_and_get_rollouts']):]] == [
        ast.dump(n) for n in b['_submit_scenarios_and_get_rollouts'].body[begin(b['_submit_scenarios_and_get_rollouts']):]]


def test_native_pool_and_completion_threshold_cancel_tail(monkeypatch):
    from phi_agents.rl import vllm_rollout_worker as owner
    from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
    cancellation = Event()
    counts, runners = {}, []
    class Runner:
        def __init__(self):
            self.closed = False
            runners.append(self)
        def run(self, scenario, llm):
            number = counts.get(scenario, 0)
            counts[scenario] = number+1
            if number >= 4:
                assert cancellation.wait(timeout=10), 'Original collector did not cancel the tail'
            return SimpleNamespace(cancelled=cancellation.is_set())
        def cleanup(self):
            self.closed = True
    monkeypatch.setattr(owner.hydra.utils, 'instantiate', lambda *a, **k: Runner())
    monkeypatch.setattr(owner, 'connect_ray_cluster', lambda *a, **k: pytest.fail('Foreign engine/Ray lifecycle invoked'))
    cfg = OmegaConf.create(dict(vllm_server=dict(gpus_per_vllm_server=1)))
    sampler = ParallelScenarioSampler(lambda: iter(range(100)))
    worker = owner.VLLMRolloutWorker(llm_cfg=cfg, scenario_sampler=sampler,
        rollouts_per_scenario=6, runner_cfg={}, rank=0, local_rank=0, barrier=lambda: None,
        inference_gpus=[0], exclusive_inference_and_learning=False, max_gpu_mem_utilization=None,
        num_runners=4, external_llm_factory=lambda event: [object()], external_cancellation_event=cancellation)
    try:
        worker.request_rollout_generation(2, None)
        scenarios, result = worker.get_rollouts(2, rollouts_fraction=2/3, rollouts_per_scenario_fraction=.75)
        assert scenarios == [0, 1]
        assert [len(r) for r in result] == [4, 4]
        assert len(runners) == 4
        assert counts == {0: 6, 1: 6}
    finally:
        worker.stop()
        sampler.stop()
    assert all(r.closed for r in runners)


@pytest.mark.parametrize('evaluation,world_size', [(False, 1), (False, 2), (True, 1)])
def test_real_appworld_process_transport(evaluation, world_size):
    """Two CPU fixture episodes, original train/eval services and token client."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from verl import DataProto
    from loop_owner_recipe import compose
    from loop_owner_worker import OwnerProcesses
    from loop_owner_rollout import LoopOwner
    from owner_environment_transport import PolicyReply
    root = Path(os.environ['LOOP_ROOT'])
    cfg = compose(root, [])
    if evaluation:
        cfg = OmegaConf.merge(cfg, cfg.rl.eval.overrides)
    cfg.rl.params.scenarios_per_iteration = world_size
    cfg.rl.params.rollouts_per_scenario = 2
    cfg.rl.num_scenario_runners = 1
    cfg.rl.rollouts_fraction = cfg.rl.rollouts_per_scenario_fraction = 1.
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    pool = OwnerProcesses(OmegaConf.to_container(cfg, resolve=True), os.environ['MODEL_PATH'], 0, evaluation, world_size)
    records, results = {}, {}
    try:
        pool.start()
        while True:
            event = pool.event()
            if event[0] == 'result':
                results[event[1]] = event[2]
                if len(results) == world_size:
                    break
                continue
            _, rank, key, request = event
            text = 'End this transport fixture.\n</think>\n```python\napis.supervisor.complete_task()\n```'
            ids = tokenizer.encode(text, add_special_tokens=False)+[tokenizer.eos_token_id]
            reply = PolicyReply(ids, [-.25]*len(ids), text, 'stop')
            prompt = request['prompt']
            from verl.utils.torch_functional import postprocess_data, pad_2d_list_to_length
            from verl.utils.model import compute_position_id_with_mask
            p, pmask = postprocess_data(torch.tensor([prompt]), torch.ones(1, len(prompt), dtype=torch.long),
                32768, tokenizer.pad_token_id, left_pad=True, truncation='error')
            response = pad_2d_list_to_length([ids], tokenizer.pad_token_id, max_length=1500)
            attention = torch.cat([pmask, (torch.arange(1500)[None, :]<len(ids)).long()], -1)
            records[key] = DataProto.from_dict(tensors=dict(input_ids=torch.cat([p, response], -1),
                prompts=p, responses=response,
                attention_mask=attention, position_ids=compute_position_id_with_mask(attention)),
                non_tensors=dict(data_source=np.array(['appworld'], dtype=object)))
            pool.replies[rank].put((key, reply))
        scenarios = [s for rank in sorted(results) for s in results[rank][0]]
        groups = [g for rank in sorted(results) for g in results[rank][1]]
        assert len(groups) == world_size and all(len(g) == 2 for g in groups)
        manager = LoopOwner.__new__(LoopOwner)
        manager.native, manager.tokenizer, manager.evaluation = cfg, tokenizer, evaluation
        rows = [(scenario, r, f'fixture-{index}') for index, (scenario, group) in enumerate(zip(scenarios, groups)) for r in group]
        batch = manager.to_batch(rows, records)
        for i, (_, rollout, _) in enumerate(rows):
            ev = rollout.appworld_rollout_data.eval_result
            expected = float(ev.success) if evaluation else len(ev.passes)/ev.num_tests
            assert rollout.ret == expected
            assert batch.batch['rm_scores'][i].sum().item() == pytest.approx(expected)
            n = len(rollout.policy_token_info.tokens)
            assert batch.batch['input_ids'][i, :n].tolist() == rollout.policy_token_info.tokens
            assert batch.batch['loss_mask'][i, :n].tolist() == rollout.policy_token_info.is_output
    finally:
        pool.close()
