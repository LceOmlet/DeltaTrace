"""Metadata boundary using the installed VERL dataset, sampler and checkpointing."""
from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import RLHFDataset


class SQLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        from sql_environment_entry import configure_sql_tokenizer
        configure_sql_tokenizer(tokenizer, config.get('sql_chat_template'))
        # Owner recipe filters initial prompts at 6000. Later tool context has a
        # separate 29000 cap; keep filtering and rollout padding distinct.
        initial = OmegaConf.merge(config, {'max_prompt_length': config.initial_max_prompt_length})
        super().__init__(data_files, tokenizer, initial, processor)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row['env_kwargs'] = dict(prompt=row['raw_prompt'], extras={
            'db_id': row['db_id'], 'data': row['data'], 'reward_spec': row['reward_spec']})
        return row
