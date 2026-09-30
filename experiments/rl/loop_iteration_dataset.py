"""Iteration triggers only; original LOOP samplers own all task selection.

VERL's dataloader requires rows to invoke its rollout interface. These rows are
never rendered or used as tasks/prompts. One dataloader batch triggers exactly
one original LOOP iteration; no second task sampler is maintained here.
"""
import torch
from torch.utils.data import Dataset


class LoopIterationDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        names = [data_files] if isinstance(data_files, str) else list(data_files)
        assert len(names) == 1 and names[0] in ('loop-training', 'loop-evaluation')
        self.size = config.loop_train_groups if names[0] == 'loop-training' else config.loop_eval_groups
        self.eos = tokenizer.eos_token_id

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return dict(input_ids=torch.tensor([self.eos]), attention_mask=torch.tensor([1]),
            position_ids=torch.tensor([0]), raw_prompt_ids=[self.eos], raw_prompt=[], env_kwargs={},
            data_source='appworld', index=index)
