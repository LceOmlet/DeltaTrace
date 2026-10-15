"""Read the installed native load/offload/forward state contract, CPU only.

The specific question is the already observed rank1 forward discrepancy in
completed same-weight/carrier replay. No model, loss, derivative or training
implementation is provided and no parameter is changed.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'

import torch
from peft import LoraConfig

def read(path, needles, radius=12):
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    intervals = []
    for i, line in enumerate(lines):
        if any(needle in line for needle in needles):
            start, stop = max(0, i - radius), min(len(lines), i + radius + 1)
            if intervals and start <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], stop)
            else:
                intervals.append([start, stop])
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        excerpts=[dict(line=start + 1, lines=lines[start:stop]) for start, stop in intervals])

files = []
for file_name, needles in (
    ('_fsdp_param.py',
     ['register_load_state_dict', 'reset_sharded_param', 'def init_unsharded_param',
      'def to_sharded', '_unsharded_param']),
    ('_fsdp_param_group.py',
     ['register_load_state_dict', 'reset_sharded_param', 'def _lazy_init',
      'def _post_forward', 'def pre_forward', 'def _reshard']),
):
    matches = list((Path(torch.__file__).parent / 'distributed').rglob(file_name))
    assert len(matches) == 1, matches
    files.append(read(matches[0], needles, radius=10))

verl = Path(os.environ['VERL_ROOT'])
files.append(read(verl / 'verl/utils/fsdp_utils.py',
    ['def load_fsdp_model_to_gpu', 'def offload_fsdp_model_to_cpu',
     'def fsdp2_load_full_state_dict', 'buf.data =', 'param.to('], radius=16))
files.append(read(verl / 'verl/workers/fsdp_workers.py',
    ['def load_checkpoint', 'def compute_log_prob', 'def update_actor',
     'load_fsdp_model_to_gpu', 'offload_fsdp_model_to_cpu'], radius=10))

from transformers.models.qwen3_5 import modeling_qwen3_5
files.append(read(inspect.getsourcefile(modeling_qwen3_5),
    ['register_buffer', 'def reset', 'rope_init_fn', 'dynamic_rope_update',
     'past_key_values', 'cache_position'], radius=8))

report = dict(unix=time.time(), scope=__doc__, torch_version=torch.__version__,
    VENV_PYTHON=os.environ['VENV_PYTHON'], VERL_ROOT=str(verl),
    native_PEFT_LoraConfig_default_dropout=LoraConfig.__dataclass_fields__['lora_dropout'].default,
    native_PEFT_LoraConfig_source=inspect.getsourcefile(LoraConfig),
    files=files, cuda_initialized=torch.cuda.is_initialized(), model_calls=0,
    backward_calls=0, optimizer_calls=0, parameter_changes=0,
    limit='A native source contract is not proof that the completed workers had identical in-memory forward state.')
assert report['cuda_initialized'] is False
path = OUT / 'native-load-forward-contract-v97.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
