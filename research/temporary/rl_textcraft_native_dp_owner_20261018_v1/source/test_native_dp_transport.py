"""CPU transport parity using the real AgentGym loop, parser and HTTP service.

Only the model replies are scripted. This tests the new placement boundary,
not model numerics, training quality, 32k capacity or performance.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from textcraft_environment_entry import configure_textcraft_tokenizer
from textcraft_native_dp_owner import collect_native_textcraft, close_native_textcraft
from textcraft_owner_rollout import TextCraftOwner
from test_vllm_active_rows import make_rollout


@pytest.fixture
def setup(monkeypatch):
    from verl.utils.debug import performance
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0., 0., 0., 0.))
    audit = Path(os.environ['NATIVE_DP_TEST_AUDIT'])
    root = audit.parents[1]
    configuration = OmegaConf.load(root / 'receipts/textcraft-eight-gpu-sequential-20261017-v3/dt-effective-config.yaml')
    configuration.env.max_steps = 2  # Explicit two-turn interface fixture only.
    tokenizer = AutoTokenizer.from_pretrained(configuration.actor_rollout_ref.model.path,
                                             local_files_only=True)
    configure_textcraft_tokenizer(tokenizer)
    path = audit / 'original/textcraft_owner_rollout.py'
    spec = importlib.util.spec_from_file_location('original_native_textcraft_owner', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return configuration, tokenizer, original.TextCraftOwner


class ScriptedReply:
    def __init__(self, tokenizer, special_tokens):
        self.tokenizer, self.special_tokens = tokenizer, special_tokens
        self.calls = []

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.calls.append(deepcopy(prompts))
        text = 'Thought: inspect.\nAction: inventory'
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if self.special_tokens:
            ids = self.tokenizer.encode('<|im_start|>', add_special_tokens=False) + ids
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(ids), text=text,
            finish_reason='stop', logprobs=[{t: SimpleNamespace(logprob=-.2)} for t in ids])])
            for _ in prompts]


def worker(tokenizer, special_tokens):
    rollout = make_rollout(vLLMRollout)
    rollout.pad_token_id = tokenizer.pad_token_id
    rollout.inference_engine = ScriptedReply(tokenizer, special_tokens)
    rollout.inference_engine.llm_engine = SimpleNamespace(dp_group=None)
    return SimpleNamespace(tokenizer=tokenizer, processor=None, rollout=rollout,
        rollout_sharding_manager=SimpleNamespace(tp_size=1), _owner_rollout_context_open=True,
        generate_sequences=rollout.generate_sequences)


@pytest.mark.parametrize('is_train,special_tokens,prompt_count', [
    (True, False, 32), (True, True, 32), (False, False, 8), (False, False, 2)])
def test_actual_owner_arrays_and_direct_target_route(setup, monkeypatch,
                                                     is_train, special_tokens, prompt_count):
    cfg, tokenizer, original = setup
    baseline = original(cfg, tokenizer, not is_train)
    candidate = TextCraftOwner(cfg, tokenizer, not is_train)
    primitive = worker(tokenizer, special_tokens)
    local_workers = [worker(tokenizer, special_tokens) for _ in range(8)]

    class Group:
        world_size = 8

        def close_native_textcraft_owners(self):
            for w in local_workers:
                close_native_textcraft(w)

        def generate_sequences(self, data):
            # The actual native DataProto partitioner and collector, with no
            # model implementation or alternate environment loop in the test.
            return DataProto.concat([collect_native_textcraft(w, part)
                for w, part in zip(local_workers, data.chunk(self.world_size))])

    data = DataProto.from_dict(tensors=dict(
        input_ids=torch.ones(prompt_count, 1, dtype=torch.long),
        attention_mask=torch.ones(prompt_count, 1, dtype=torch.long),
        position_ids=torch.zeros(prompt_count, 1, dtype=torch.long)),
        non_tensors=dict(env_kwargs=np.array(
            [dict(item_id=31 + i % 2) for i in range(prompt_count)], dtype=object),
            data_source=np.array(['TextCraft'] * prompt_count, dtype=object)),
        meta_info=dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
    initial_metadata = deepcopy(data.meta_info)
    collector = TrajectoryCollector(cfg, tokenizer)
    try:
        expected = baseline.collect_native_trajectories(collector, data,
            SimpleNamespace(world_size=8, generate_sequences=primitive.generate_sequences), is_train)
        actual = candidate.collect_native_trajectories(collector, data, Group(), is_train)
        assert data.meta_info == initial_metadata
        assert expected.batch.keys() == actual.batch.keys()
        for key in expected.batch.keys():
            assert torch.equal(expected.batch[key], actual.batch[key]), key
        assert expected.non_tensor_batch.keys() == actual.non_tensor_batch.keys()
        for key in expected.non_tensor_batch:
            a, b = expected.non_tensor_batch[key], actual.non_tensor_batch[key]
            if key in ('uid', 'traj_uid'):
                assert [[v == a[0] for v in a]] == [[v == b[0] for v in b]]
                assert len(set(a)) == len(set(b))
            else:
                assert a.tolist() == b.tolist(), key
        native = candidate.native['train' if is_train else 'eval']
        n = native['actor_rollout_ref']['rollout']['n'] if is_train else native['rollout']['n']
        assert len(actual) == prompt_count * n
        assert len(set(actual.non_tensor_batch['uid'])) == prompt_count
        assert all(len(set(actual.non_tensor_batch['uid'][i:i+n])) == 1
                   for i in range(0, len(actual), n))
        assert len(set(actual.non_tensor_batch['traj_uid'])) == len(actual)
        assert all(len(w.rollout.inference_engine.calls) == 2 for w in local_workers)
        if cfg.algorithm.adv_estimator == 'deltatrace':
            from owner_trajectory_batch import trajectory_credit
            import dt_training_batch
            sentinel, seen = object(), []

            def direct(data, group, **kwargs):
                seen.append(data)
                return sentinel

            monkeypatch.setattr(dt_training_batch, 'compute_direct_target_credit', direct)
            assert candidate.credit_responses is None
            assert trajectory_credit(actual, candidate.credit_responses, Group(),
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id) is sentinel
            assert len(seen) == 1 and seen[0] is actual
    finally:
        baseline.close()
        candidate.close()
        assert all(getattr(w, '_native_textcraft_episode_owners', None) is None for w in local_workers)
        for w in local_workers:
            if getattr(w, '_native_textcraft_episode_owners', None) is not None:
                train, evaluation, _ = w._native_textcraft_episode_owners
                train.close()
                evaluation.close()
