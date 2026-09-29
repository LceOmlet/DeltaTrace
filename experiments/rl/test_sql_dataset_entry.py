"""Real official parquet and Qwen tokenizer through the original VERL dataset."""
import os
from pathlib import Path
import hashlib
import json

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from owner_task_dataset import SQLDataset


def test_official_data_filters_by_token_ids_and_keeps_environment_metadata():
    config = OmegaConf.load(os.environ['DT_SQL_CONFIG'])
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    dataset = SQLDataset(config.data.train_files, tokenizer, config.data)
    assert len(dataset) > 0
    # Compare to the exact native filter predicate on this model's actual IDs.
    import pyarrow.parquet as pq
    original = pq.read_table(config.data.train_files).to_pylist()
    accepted = [r for r in original if len(tokenizer.apply_chat_template(
        r['prompt'], add_generation_prompt=True, return_dict=False)) <= 6000]
    assert len(dataset) == len(accepted)
    if config.data.get('sql_chat_template'):
        probe = tokenizer.apply_chat_template([dict(role='user',content='SQL')],
            add_generation_prompt=True, tokenize=False)
        assert probe.endswith('<|im_start|>assistant\n')
        assert not probe.endswith('<think>\n')
    for index in [0, len(dataset)-1]:
        row = dataset[index]
        assert row['env_kwargs']['prompt'] == row['raw_prompt']
        assert row['env_kwargs']['extras'] == {k: accepted[index][k] for k in ('db_id', 'data', 'reward_spec')}
        assert row['attention_mask'].sum().item() == len(row['raw_prompt_ids'])
    from hydra.utils import instantiate
    train, evaluation = instantiate(config.env.factory, configuration=config, tokenizer=tokenizer, _recursive_=False)
    for manager in [train, evaluation]:
        try:
            observation, _ = manager.reset([dataset[0]['env_kwargs']])
            assert observation['raw_prompt_ids'][0] == dataset[0]['raw_prompt_ids']
        finally:
            manager.close()


def test_selected_databases_are_the_unmodified_official_archive_members():
    config = OmegaConf.load(os.environ['DT_SQL_CONFIG'])
    root = Path(config.env.sql.db_path).parent
    manifest = json.loads((root/'selected-databases.json').read_text())
    # Manifest's exact schema is produced by the official archive acquisition.
    assert len(manifest) == 611
    for item in manifest:
        file = root/item['name']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['sha256']


def test_native_collector_consumes_engine_artifacts_and_official_terminal_reward(monkeypatch):
    # Scripted output isolates the environment/collector interface; no claim of
    # model quality. Real data, original SQLite reward, collector and vLLM
    # transport are executed, with only LLM.generate replaced by this fixture.
    from types import SimpleNamespace
    import torch
    from hydra.utils import instantiate
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.debug import performance
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from test_vllm_active_rows import make_rollout
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0., 0., 0., 0.))
    config = OmegaConf.load(os.environ['DT_SQL_CONFIG'])
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    dataset = SQLDataset(config.data.train_files, tokenizer, config.data)
    row = dataset[0]
    text = '<think>Interface fixture.</think><solution>' + row['reward_spec']['ground_truth'] + '</solution>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    class Engine:
        def generate(self, *, prompts, sampling_params, **kwargs):
            from vllm import SamplingParams
            expected = SamplingParams.from_optional(**OmegaConf.to_container(config.env.sql.sampling))
            assert all(p == expected for p in sampling_params)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, text=text, finish_reason='stop',
                logprobs=[{t: SimpleNamespace(logprob=-.2)} for t in ids])]) for _ in prompts]
    rollout = make_rollout(vLLMRollout)
    rollout.pad_token_id = tokenizer.pad_token_id
    rollout.inference_engine = Engine()
    group = SimpleNamespace(world_size=2, generate_sequences=rollout.generate_sequences)
    batch = DataProto.from_single_dict(collate_fn([row, dataset[0]]))
    batch.meta_info = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    train, _ = instantiate(config.env.factory, configuration=config, tokenizer=tokenizer, _recursive_=False)
    try:
        rows, rewards, lengths, _, _, _ = TrajectoryCollector(config, tokenizer).vanilla_multi_turn_loop(batch, group, train)
        assert rewards.tolist() == [1., 1.] and lengths.tolist() == [1., 1.]
        for episode in rows:
            assert len(episode) == 1
            result = episode[0]
            assert result['responses'][:len(ids)].tolist() == ids
            assert result['attention_mask'][-1024:].sum().item() == len(ids)
            torch.testing.assert_close(result['rollout_log_probs'][:len(ids)], torch.full((len(ids),), -.2))
    finally:
        train.close()
