"""CPU checks against actual owner handler, service and original VERL collector."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers import AutoTokenizer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from launch_owner_entry_check import options_for
from owner_runtime_options import owner_command
from textcraft_environment_entry import owner_module
from test_vllm_active_rows import make_rollout


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)


def handler(cls):
    return cls(messages=[], task_name='textcraft', item_id=31, score=0, done=False,
        input_ids=[248046], prompt_ids=[248046], response_ids=[], attention_mask=[1],
        prompt_attention_mask=[1], response_attention_mask=[], position_ids=[0],
        prompt_position_ids=[0], response_position_ids=[], loss_mask=[0],
        prompt_loss_mask=[0], response_loss_mask=[], max_response_len=10240, max_model_len=10752)


def test_default_handler_is_unchanged(tokenizer):
    from functools import partial
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, return_dict=False)
    original = owner_module(Path(os.environ['TEXTCRAFT_BASELINE_SCHEMA']), 'textcraft_unmodified_schema')
    modified = owner_module(Path(os.environ['AGENTGYM_RL_ROOT'])/'AgentGym-RL/verl/workers/rollout/schemas.py',
                            'textcraft_modified_schema')
    a, b = handler(original.RolloutHandler), handler(modified.RolloutHandler)
    for h in (a, b):
        h.add_user_message(tokenizer, 'Craft blue dye.')
        h.add_assistant_message(tokenizer, 'Thought: collect.\nAction: get 1 lapis lazuli')
        h.add_user_message(tokenizer, 'Got 1 lapis lazuli.')
        h.add_assistant_message(tokenizer, 'Thought: craft.\nAction: craft 1 blue dye using 1 lapis lazuli')
        h.truncate_output_ids()
    assert a.get_generation_prompt(tokenizer) == b.get_generation_prompt(tokenizer)
    for key in vars(a):
        if key == 'messages':
            assert [m.to_dict() for m in a.messages] == [m.to_dict() for m in b.messages]
        else:
            assert getattr(a, key) == getattr(b, key), key


@pytest.mark.parametrize('is_train', [True, False])
@pytest.mark.parametrize('failure', [None, 'step', 'reset', 'special_tokens'])
def test_real_service_native_trajectory_and_exact_generated_prefix(tokenizer, tmp_path, monkeypatch, is_train, failure):
    from verl.utils.debug import performance
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0., 0., 0., 0.))
    options, _ = options_for('TextCraft', tmp_path, tmp_path)
    options['actor_rollout_ref.rollout.multi_turn.enable'] = True
    with initialize_config_dir(config_dir=str(Path(os.environ['VERL_ROOT'])/'verl/trainer/config'),version_base=None):
        config = compose(config_name='ppo_trainer', overrides=owner_command(options)[3:])
    # A two-interaction interface fixture, not a new experimental configuration.
    config.env.max_steps = 2
    train, evaluation = instantiate(config.env.factory, configuration=config, tokenizer=tokenizer, _recursive_=False)
    manager = train if is_train else evaluation
    if failure in ('step', 'reset'):
        original_client = manager.module.init_env_client
        def failing_client(*args, **kwargs):
            client = original_client(*args, **kwargs)
            def fail(*a, **kw):
                raise TimeoutError('scripted owner error-path fixture')
            setattr(client, failure, fail)
            return client
        monkeypatch.setattr(manager.module, 'init_env_client', failing_client)
    collector = TrajectoryCollector(config, tokenizer)
    class Engine:
        calls = []
        def generate(self, *, prompts, sampling_params, **kwargs):
            self.calls.append(deepcopy(prompts))
            text = 'Thought: inspect.\nAction: inventory'
            ids = tokenizer.encode(text, add_special_tokens=False)+[tokenizer.eos_token_id]
            if failure == 'special_tokens':
                ids = tokenizer.encode('<|im_start|>', add_special_tokens=False)+ids
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids.copy(),text=text,
                finish_reason='stop',logprobs=[{t:SimpleNamespace(logprob=-.2)} for t in ids])]) for _ in prompts]
    rollout = make_rollout(vLLMRollout)
    rollout.pad_token_id = tokenizer.pad_token_id
    rollout.inference_engine = Engine()
    group = SimpleNamespace(world_size=2,generate_sequences=rollout.generate_sequences)
    batch = DataProto.from_dict(tensors=dict(input_ids=torch.ones(2,1,dtype=torch.long),
        attention_mask=torch.ones(2,1,dtype=torch.long),position_ids=torch.zeros(2,1,dtype=torch.long)),
        non_tensors=dict(raw_prompt=np.array([[dict(role='user',content='item')]]*2,dtype=object),
            env_kwargs=np.array([dict(item_id=31),dict(item_id=32)],dtype=object),
            data_source=np.array(['TextCraft']*2,dtype=object)))
    batch.meta_info.update(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    try:
        output = collector.multi_turn_loop(batch,group,manager,is_train=is_train)
        assert len(output) == (16 if is_train else 2)
        rounds = 0 if failure == 'reset' else 1 if failure == 'step' else 2
        assert len(rollout.inference_engine.calls) == rounds
        assert output.batch['input_ids'].shape[-1] == (10752 if is_train else 14848)
        for row, h in enumerate(manager.handlers):
            for step, record in enumerate(manager.records[row]):
                prompt = rollout.inference_engine.calls[step][row]['prompt_token_ids']
                assert h.input_ids[:record['start']] == prompt
                assert h.input_ids[record['start']:record['end']] == record['native_response_ids']
                if failure == 'special_tokens':
                    assert record['native_response_ids'] != record['output_ids']
                else:
                    assert record['native_response_ids'] == record['output_ids']
            torch.testing.assert_close(output.batch['response_mask'][row,:len(h.response_loss_mask)],
                torch.tensor(h.response_loss_mask,dtype=output.batch['response_mask'].dtype),rtol=0,atol=0)
            assert output.batch['rm_scores'][row].sum().item() == h.score
        assert not compute_response_mask(output)[output.batch['attention_mask'][:,-output.batch['responses'].shape[-1]:]==0].any()
        assert (0 if manager.credit_responses is None else len(manager.credit_responses)) == len(output)*rounds
        # Nonzero sentinels test only credit transport, not DT's estimates.
        import dt_training_batch
        from owner_trajectory_batch import trajectory_credit
        if rounds:
            shape = manager.credit_responses.batch['responses'].shape
            markers = torch.arange(1, shape[0]+1, dtype=torch.float32)[:, None].expand(shape).clone()
            monkeypatch.setattr(dt_training_batch, 'compute_training_credit', lambda *a, **k:
                DataProto.from_dict(tensors={key: markers for key in dt_training_batch.CREDIT_KEYS}))
        credit = trajectory_credit(output, manager.credit_responses, group,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        for key in dt_training_batch.CREDIT_KEYS:
            assert not credit.batch[key][compute_response_mask(output) == 0].any()
            for row, slices in enumerate(output.non_tensor_batch['dt_response_slices']):
                for source, start, length in slices:
                    assert (credit.batch[key][row, start:start+length] == source+1).all()
        if failure in ('step', 'reset'):
            assert not output.batch['rm_scores'].any()
            assert all(h.done for h in manager.handlers)
    finally:
        train.close()
        evaluation.close()


def test_original_rollout_algorithm_is_not_reimplemented():
    import ast
    original = Path(os.environ['TEXTCRAFT_ORIGINAL_ROLLOUT']).read_text()
    current = (Path(os.environ['AGENTGYM_RL_ROOT'])/'AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py').read_text()
    # Only progress/logging can run without a local distributed process group.
    current = current.replace('(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)',
                              'torch.distributed.get_rank()')
    def methods(source):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'vLLMRollout')
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    a, b = methods(original), methods(current)
    for name in ['preprocess_prompt_to_rollout_handler', 'generate_sequences', 'update_sampling_params']:
        assert a[name] == b[name], name
