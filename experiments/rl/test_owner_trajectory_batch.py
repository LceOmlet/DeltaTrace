"""CPU environment/token transport through the actual native collector.

Only LLM.generate is a scripted fixture. SQL execution, complete trajectory
assembly, masks, native batch containers and DT index transport are real.
"""
from types import SimpleNamespace
from pathlib import Path
import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

from launch_sql_native import command
from test_sql_owner_rollout import tokenizer, task
from test_vllm_active_rows import make_rollout


@pytest.mark.parametrize('method', ['dt', 'grpo'])
def test_native_trajectory_is_one_training_row_and_observations_are_masked(
        tokenizer, task, tmp_path, monkeypatch, method):
    from verl.utils.debug import performance
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0., 0., 0., 0.))
    argv, _ = command(SimpleNamespace(method=method, phase='formal', data=tmp_path, output=tmp_path))
    with initialize_config_dir(config_dir=str(Path(os.environ['VERL_ROOT'])/'verl/trainer/config'),version_base=None):
        config = compose(config_name='ppo_trainer', overrides=argv[3:])
    config.env.sql.db_path = str(tmp_path)
    train, evaluation = instantiate(config.env.factory, configuration=config, tokenizer=tokenizer, _recursive_=False)
    collector = TrajectoryCollector(config, tokenizer)
    replies = [
        '<think>Inspect.</think><sql>select name from items</sql>',
        '<think>Answer.</think><solution>select name from items</solution>',
    ]
    class Engine:
        calls = 0
        def generate(self, *, prompts, sampling_params, **kwargs):
            text = replies[self.calls]
            self.calls += 1
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=ids.copy(), text=text, finish_reason='stop',
                logprobs=[{t:SimpleNamespace(logprob=-.2)} for t in ids])]) for _ in prompts]
    rollout = make_rollout(vLLMRollout)
    rollout.pad_token_id = tokenizer.pad_token_id
    rollout.inference_engine = Engine()
    group = SimpleNamespace(world_size=2, generate_sequences=rollout.generate_sequences)
    # Two real SQLite episodes, with the native recipe's five repetitions.
    batch = DataProto.from_dict(tensors=dict(input_ids=torch.ones(2,1,dtype=torch.long),
        attention_mask=torch.ones(2,1,dtype=torch.long),position_ids=torch.zeros(2,1,dtype=torch.long)),
        non_tensors=dict(raw_prompt=np.array([task['prompt']]*2,dtype=object),
                        env_kwargs=np.array([task]*2,dtype=object),
                        data_source=np.array(['text2sql']*2,dtype=object)))
    batch.meta_info.update(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    try:
        output = collector.multi_turn_loop(batch, group, train)
        assert len(output) == 10  # not twenty response rows
        assert rollout.inference_engine.calls == 2
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 1280
        assert config.actor_rollout_ref.actor.ppo_epochs == 1
        assert all(float(x) == 1. for x in output.batch['rm_scores'].sum(-1))
        # The native batch converter declares/returns float32 loss masks.
        expected = torch.tensor(train.current[0].trajectory.loss_mask,dtype=torch.float32)
        torch.testing.assert_close(output.batch['loss_mask'][0,-len(expected):],expected,rtol=0,atol=0)
        response_mask = compute_response_mask(output)
        assert (response_mask == 0).any()  # SQLite observation is context only.
        torch.testing.assert_close(response_mask[0,-len(expected):],expected,rtol=0,atol=0)
        assert len(set(output.non_tensor_batch['uid'])) == 2
        if method == 'dt':
            assert len(train.credit_responses) == 20
            import dt_training_batch
            from owner_trajectory_batch import trajectory_credit
            from dt_training_batch import CREDIT_KEYS
            source_shape = train.credit_responses.batch['responses'].shape
            fixtures = {key: torch.arange(source_shape[0],dtype=torch.float32)[:,None].expand(source_shape).clone()+1
                        for key in CREDIT_KEYS}
            monkeypatch.setattr(dt_training_batch,'compute_training_credit',lambda *a,**kw:DataProto.from_dict(tensors=fixtures))
            # Native balancing/adjust_batch may reorder/duplicate trajectories.
            arranged = output.select_idxs([9,0,3,9])
            credits = trajectory_credit(arranged,train.credit_responses,group,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
            for row,slices in enumerate(arranged.non_tensor_batch['dt_response_slices']):
                for source,start,length in slices:
                    for key in CREDIT_KEYS:
                        torch.testing.assert_close(credits.batch[key][row,start:start+length],fixtures[key][source,:length])
                for key in CREDIT_KEYS:
                    assert not credits.batch[key][row][compute_response_mask(arranged)[row]==0].any()
    finally:
        train.close()
        evaluation.close()
