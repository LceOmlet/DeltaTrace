"""Exact completion transport through original VERL DP split/collect on CPU.

The fixture supplies queued owner requests and engine replies only. It does
not emulate the environment, scheduling, sampling, reward, or PPO algorithm.
"""
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.base.decorator import (
    dispatch_dp_compute_data_proto, collect_dp_compute_data_proto,
)

from loop_owner_rollout import LoopOwner


class NativeDispatch(WorkerGroup):
    def __init__(self):
        self._workers = [None, None]
        self.calls = []

    def generate_sequences(self, batch):
        args, kwargs = dispatch_dp_compute_data_proto(self, batch)
        assert kwargs == {}
        self.calls.append([len(chunk) for chunk in args[0]])
        outputs = []
        for chunk in args[0]:
            ids = chunk.batch['input_ids'].clone()
            outputs.append(DataProto.from_dict(
                tensors=dict(responses=ids, rollout_log_probs=torch.full_like(ids, -.25, dtype=torch.float)),
                non_tensors=dict(owner_response_length=np.ones(len(chunk), dtype=int),
                    owner_response_text=np.array([str(i.item()) for i in ids[:, 0]], dtype=object),
                    owner_finish_reason=np.array(['stop'] * len(chunk), dtype=object))))
        return collect_dp_compute_data_proto(self, outputs)


@pytest.mark.parametrize('count,cancel_rank', [(64, None), (63, None), (64, 0)])
def test_all_queued_owner_requests_reach_original_dp_dispatch(count, cancel_rank):
    class Pool:
        def __init__(self):
            self.processes = [object(), object()]
            self.output = Queue()
            self.replies = [Queue(), Queue()]
            self.cancellations = [Event(), Event()]
            if cancel_rank is not None:
                self.cancellations[cancel_rank].set()
            for i in range(count):
                self.output.put(('completion', i % 2, str(i), dict(
                    prompt=[i+1], model='fixture', stream=False, max_tokens=1500)))
            for rank in range(2):
                self.output.put(('result', rank, ([], [])))

        def start(self):
            pass

        def event(self):
            return self.output.get(timeout=2)

        def queued_event(self):
            return self.output.get_nowait()

    pool = Pool()
    observed_options = []
    def preprocess(carrier, obs):
        observed_options.extend(obs['sampling_kwargs'])
        return DataProto.from_dict(tensors=dict(input_ids=torch.tensor(obs['raw_prompt_ids'])))

    manager = LoopOwner.__new__(LoopOwner)
    manager.processes = pool
    manager.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        rollout=SimpleNamespace(max_num_seqs=32)))
    manager.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    manager.to_batch = lambda rows, records: records
    group = NativeDispatch()
    records = manager.collect_native_trajectories(
        SimpleNamespace(preprocess_batch=preprocess), SimpleNamespace(meta_info={}), group, True)

    active = [i for i in range(count) if i % 2 != cancel_rank]
    expected_per_rank = (len(active)+1)//2
    assert group.calls == [[expected_per_rank, expected_per_rank]]
    assert set(records) == {str(i) for i in active}
    assert len(observed_options) == len(active)
    assert all(o == dict(max_tokens=1500, detokenize=True) for o in observed_options)
    for rank, queue in enumerate(pool.replies):
        expected = [i for i in active if i % 2 == rank]
        assert queue.qsize() == len(expected)
        for i in expected:
            key, reply = queue.get_nowait()
            assert key == str(i)
            assert reply.token_ids == [i+1]
            assert reply.logprobs == [-.25]
            assert reply.text == str(i+1)
            assert reply.finish_reason == 'stop'
    assert manager.config.actor_rollout_ref.rollout.max_num_seqs == 32
