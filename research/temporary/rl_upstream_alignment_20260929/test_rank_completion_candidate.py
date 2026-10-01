"""CPU transport tests using real Ray ActorPool and pinned VERL RPC helpers.

The actor supplies known engine reply artifacts only. There is no model,
environment, sampler, scheduler, reward or optimizer substitute in the tests.
"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import os
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from unittest.mock import Mock
from verl import DataProto
from verl.single_controller.base.decorator import (
    collect_dp_compute_data_proto, dispatch_dp_compute_data_proto,
)
from verl.single_controller.ray.base import RayWorkerGroup, func_generator


@ray.remote(num_cpus=0)
class ReplyActor:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.released = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = []

    async def release(self):
        self.released.set()

    async def snapshot(self):
        return self.calls

    async def wait_started(self):
        await self.started.wait()

    async def actor_rollout_generate_sequences(self, batch):
        assert len(batch), 'The original VERL generator does not support B0'
        self.calls.append(batch.batch['input_ids'][:, 0].tolist())
        self.started.set()
        if self.blocked:
            await self.released.wait()
        ids = batch.batch['input_ids'].clone()
        if (ids == -1).any():
            raise RuntimeError('engine fixture failure')
        return DataProto.from_dict(
            tensors=dict(responses=ids, rollout_log_probs=torch.full_like(ids, -.25, dtype=torch.float)),
            non_tensors=dict(batch.non_tensor_batch,
                owner_response_length=np.ones(len(batch), dtype=int),
                owner_response_text=np.array([str(i.item()) for i in ids[:, 0]], dtype=object),
                owner_finish_reason=np.array(['stop']*len(batch), dtype=object)))


@pytest.fixture(scope='module', autouse=True)
def cpu_ray():
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False,
             object_store_memory=80*2**20, _memory=512*2**20,
             _temp_dir=f'/tmp/rc-{os.getpid()}')
    yield
    ray.shutdown()


def owner_group(actors):
    # The minimal initialized fields for the unchanged owner RPC helper.
    # The actual Functor and wire-method mapping are built by VERL itself.
    group = RayWorkerGroup.__new__(RayWorkerGroup)
    group._workers = actors
    group._world_size = len(actors)
    group.fused_worker_used = False
    group.method_names = []
    group.generate_sequences = func_generator(group, 'actor_rollout_generate_sequences',
        dispatch_dp_compute_data_proto, collect_dp_compute_data_proto,
        group.execute_all_async, True)
    return group


def candidate():
    path = Path(__file__).with_name('appworld-rank-completion-20261002')/'loop_owner_rollout.py'
    spec = importlib.util.spec_from_file_location('rank_completion_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoopOwner


class Requests:
    def __init__(self):
        self.processes = [object(), object()]
        self.output = Queue()
        self.replies = [Queue(), Queue()]
        self.cancellations = [Event(), Event()]

    def start(self):
        pass

    def event(self):
        return self.output.get(timeout=15)

    def queued_event(self):
        return self.output.get_nowait()

    def request(self, rank, key, token):
        self.output.put(('completion', rank, key, dict(prompt=[token],
            model='fixture', stream=False, max_tokens=1500)))

    def finish(self, rank):
        self.output.put(('result', rank, ([], [])))


def collect(requests, group, options):
    def preprocess(carrier, obs):
        options.extend(obs['sampling_kwargs'])
        return DataProto.from_dict(tensors=dict(input_ids=torch.tensor(obs['raw_prompt_ids'])))
    cls = candidate()
    manager = cls.__new__(cls)
    manager.processes = requests
    manager.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    manager.to_batch = lambda rows, records: records
    return manager.collect_native_trajectories(SimpleNamespace(preprocess_batch=preprocess),
        SimpleNamespace(meta_info={}), group, True)


@pytest.mark.parametrize('count,cancel_rank', [(64, None), (63, None), (64, 0), (1, None)])
def test_actual_owner_pool_preserves_exact_requests_and_replies(count, cancel_rank):
    requests = Requests()
    if cancel_rank is not None:
        requests.cancellations[cancel_rank].set()
    for i in range(count):
        requests.request(i % 2, str(i), i+1)
    for rank in range(2):
        requests.finish(rank)
    actors = [ReplyActor.remote(), ReplyActor.remote()]
    options = []
    try:
        records = collect(requests, owner_group(actors), options)
        active = [i for i in range(count) if i % 2 != cancel_rank]
        assert set(records) == {str(i) for i in active}
        generated = [value for calls in ray.get([a.snapshot.remote() for a in actors])
                     for call in calls for value in call]
        assert sorted(generated) == [i+1 for i in active]
        assert options == [dict(max_tokens=1500, detokenize=True)]*len(active)
        for rank, queue in enumerate(requests.replies):
            expected = {str(i):i+1 for i in active if i % 2 == rank}
            assert queue.qsize() == len(expected)
            while not queue.empty():
                key, reply = queue.get_nowait()
                assert reply.token_ids == [expected.pop(key)]
                assert reply.logprobs == [-.25]
                assert reply.text == str(reply.token_ids[0])
                assert reply.finish_reason == 'stop'
            assert not expected
    finally:
        for actor in actors:
            ray.kill(actor)


def test_fast_rank_can_advance_again_before_slow_rank_returns():
    requests = Requests()
    # ActorPool takes the last idle actor first. Keep the first chunk blocked.
    actors = [ReplyActor.remote(), ReplyActor.remote(blocked=True)]
    requests.request(0, 'slow', 101)
    requests.request(1, 'fast', 202)
    options = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(collect, requests, owner_group(actors), options)
        try:
            assert requests.replies[1].get(timeout=15)[0] == 'fast'
            assert requests.replies[0].empty()
            requests.request(1, 'next', 303)
            assert requests.replies[1].get(timeout=15)[0] == 'next'
            assert not future.done()  # Native scope must drain the slow RPC.
            requests.finish(1)
            ray.get(actors[1].release.remote())
            assert requests.replies[0].get(timeout=15)[0] == 'slow'
            requests.finish(0)
            records = future.result(timeout=15)
            assert set(records) == {'slow', 'fast', 'next'}
        finally:
            ray.get(actors[1].release.remote())
            requests.finish(0)
            requests.finish(1)
            for actor in actors:
                ray.kill(actor)


def test_original_engine_error_is_not_hidden():
    requests = Requests()
    requests.request(0, 'error', -1)
    requests.finish(0)
    requests.finish(1)
    actors = [ReplyActor.remote(), ReplyActor.remote()]
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match='engine fixture failure'):
            collect(requests, owner_group(actors), [])
    finally:
        for actor in actors:
            ray.kill(actor)


def test_native_cancellation_is_rechecked_after_owner_pool_queueing():
    requests = Requests()
    actors = [ReplyActor.remote(blocked=True), ReplyActor.remote(blocked=True)]
    requests.request(0, 'first', 1)
    requests.request(1, 'second', 2)
    prepared = Event()
    class ObservedOptions(list):
        def extend(self, values):
            super().extend(values)
            if len(self) >= 4:
                prepared.set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(collect, requests, owner_group(actors), ObservedOptions())
        try:
            ray.get([a.wait_started.remote() for a in actors], timeout=15)
            requests.request(0, 'cancelled', 3)
            requests.request(1, 'live', 4)
            assert prepared.wait(15)
            requests.cancellations[0].set()
            ray.get([a.release.remote() for a in actors])
            requests.finish(0)
            requests.finish(1)
            records = future.result(timeout=15)
            assert set(records) == {'first', 'second', 'live'}
            generated = [value for calls in ray.get([a.snapshot.remote() for a in actors])
                         for call in calls for value in call]
            assert sorted(generated) == [1, 2, 4]
        finally:
            ray.get([a.release.remote() for a in actors])
            requests.finish(0)
            requests.finish(1)
            for actor in actors:
                ray.kill(actor)


def test_original_verl_empty_input_is_not_a_supported_rpc_contract():
    from omegaconf import OmegaConf
    from vllm import SamplingParams
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    owner = vLLMRollout.__new__(vLLMRollout)
    owner.config = OmegaConf.create(dict(free_cache_engine=False, response_length=32767))
    owner.pad_token_id = 0
    owner.lora_kwargs = {}
    owner.sampling_params = SamplingParams(n=1, logprobs=0)
    owner.inference_engine = Mock()
    owner.inference_engine.generate.side_effect = AssertionError('Empty transport must not generate')
    batch = DataProto.from_dict(tensors=dict(
        input_ids=torch.empty((0, 1), dtype=torch.long),
        attention_mask=torch.empty((0, 1), dtype=torch.long),
        position_ids=torch.empty((0, 1), dtype=torch.long)),
        non_tensors=dict(raw_prompt_ids=np.array([], dtype=object),
            owner_sampling_kwargs=np.array([], dtype=object),
            loop_reply_key=np.array([], dtype=object), loop_reply_rank=np.array([], dtype=int)),
        meta_info=dict(eos_token_id=248044, pad_token_id=0))
    # Only the GPU memory logger is removed for this CPU-only body test.
    # The original generation implementation, helpers and result are unchanged.
    from verl.utils.debug.performance import GPUMemoryLogger
    wrapper = inspect.getclosurevars(vLLMRollout.generate_sequences).nonlocals
    assert isinstance(wrapper['self'], GPUMemoryLogger)
    with pytest.raises(ValueError, match='max\\(\\) iterable argument is empty'):
        wrapper['decorated_function'](owner, batch)
    owner.inference_engine.generate.assert_not_called()


def test_candidate_only_changes_completion_transport():
    before = Path(os.environ['ORIGINAL_LOOP_ENTRY'])/'loop_owner_rollout.py'
    after = Path(__file__).with_name('appworld-rank-completion-20261002')/'loop_owner_rollout.py'
    def methods(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoopOwner')
        return {n.name:ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    original, patched = methods(before), methods(after)
    assert original.keys() == patched.keys()
    assert {k for k in original if original[k] != patched[k]} == {'collect_native_trajectories'}
