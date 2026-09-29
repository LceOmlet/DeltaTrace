"""Exercise the original trainer's supported single-process metadata loader."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

audit = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/upstream-alignment-20260929'
cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
directory = audit/'two-gpu-pilots-isolated/dt-AppWorld'
for key, value in {
    'data.train_files': [str(directory/'data/text/train.parquet')],
    'data.val_files': [str(directory/'data/text/test.parquet')],
    'data.train_batch_size': 4, 'data.val_batch_size': 1,
    'data.max_prompt_length': 32019, 'data.max_response_length': 512,
    'data.truncation': 'error', 'data.return_raw_chat': True,
    'data.apply_chat_template_kwargs': {'enable_thinking': False},
    'data.dataloader_num_workers': 0, 'trainer.total_epochs': 3,
}.items():
    OmegaConf.update(cfg, key, value, force_add=True)
tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)

def trainer():
    obj = object.__new__(RayPPOTrainer)
    obj.config = copy.deepcopy(cfg)
    obj.tokenizer, obj.processor = tokenizer, None
    obj._create_dataloader(None, None, None, None)
    return obj

def same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if isinstance(a[name], torch.Tensor):
            assert torch.equal(a[name], b[name]), name
        else:
            assert a[name].tolist() == b[name].tolist(), name

# Reproduce the parent-side initialized CPU threadpool, without forking another
# worker from it. Dataset, sampler, collator and state transitions stay upstream.
torch.set_num_threads(8)
torch.mm(torch.ones(1024, 1024), torch.ones(1024, 1024))
native = trainer()
times, counts = [], []
for epoch in range(3):
    tick = time.perf_counter()
    batches = list(native.train_dataloader)
    times.append(time.perf_counter()-tick)
    counts.append(sum(len(b['input_ids']) for b in batches))
    assert len(batches) == 1 and counts[-1] == 4
    assert all(torch.isfinite(b['input_ids']).all() for b in batches)
snapshot = native.train_dataloader.state_dict()
expected = list(native.train_dataloader)
resumed = trainer()
resumed.train_dataloader.load_state_dict(snapshot)
actual = list(resumed.train_dataloader)
assert len(actual) == len(expected)
resume_differences = []
for left, right in zip(actual, expected):
    try:
        same(left, right)
    except AssertionError as exc:
        resume_differences.append(str(exc))
result = dict(scope=__doc__, owner=str(Path(os.environ['VERL_ROOT'])/'verl/trainer/ppo/ray_trainer.py'),
    official_parameter='data.dataloader_num_workers', value=0, epochs=3, rows=counts,
    seconds=times, native_checkpoint_resume_exact=not resume_differences,
    native_checkpoint_resume_differences=resume_differences,
    note='New-run configuration. Existing eight-worker data.pt is a different upstream format; do not silently reset or convert it.')
(audit/'native-single-process-dataloader.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
