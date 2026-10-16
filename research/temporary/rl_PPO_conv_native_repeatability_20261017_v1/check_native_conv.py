"""Bounded native Conv1d repeatability check, not PPO numerical acceptance.

Use the installed torch owner, actual Qwen3.5 convolution dimensions and BF16
autocast. Synthetic fixed input/upstream gradient isolate kernel repeatability;
no model, optimizer, credit assignment or formal training change is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/PPO-conv-native-repeatability-20261017-v1'
if '--native' not in sys.argv:
    smi = subprocess.check_output(['mx-smi'], text=True)
    assert not any(re.match(r'\|\s*2\s+\d+\s+\S+', line) for line in smi.splitlines()), 'Physical GPU2 is occupied'
    prepared = json.loads((ROOT / 'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json').read_bytes())
    env = dict(os.environ)
    env.update(prepared['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = '2'
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--native'], env=env))

import inspect
import psutil
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig

owner = ROOT / 'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
cfg = OmegaConf.load(owner / 'verl/trainer/config/ppo_trainer.yaml')
prepared = json.loads((ROOT / 'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
for key, value in prepared['options'].items():
    OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
model_cfg = AutoConfig.from_pretrained(cfg.critic.model.path, trust_remote_code=True, local_files_only=True)
text_cfg = model_cfg.text_config
channels = 2 * text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
width = text_cfg.linear_conv_kernel_dim
torch.manual_seed(20261017)
conv = torch.nn.Conv1d(channels, channels, width, groups=channels, padding=width - 1,
                       bias=False, device='cuda', dtype=torch.float32)
x = torch.randn(4, channels, 8192, dtype=torch.bfloat16, device='cuda', requires_grad=True)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = conv(x)
upstream = torch.randn_like(output)
del output
original = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=torch.backends.cudnn.allow_tf32)
receipt = dict(version='PPO-conv-native-repeatability-20261017-v1', started=time.time(),
               pid=os.getpid(), birth=psutil.Process().create_time(), physical_gpus=[2],
               source_commit=os.environ['PPO_CONV_SOURCE_COMMIT'],
               source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               torch_version=torch.__version__, Conv1d_owner=inspect.getfile(torch.nn.Conv1d),
               cudnn_available=torch.backends.cudnn.is_available(), cudnn_version=torch.backends.cudnn.version(),
               input_shape=list(x.shape), kernel_width=width, groups=channels,
               original_flags=original, model_calls=0, optimizer_updates=0, formal_training_mutations=0,
               scope=__doc__, modes=[])
try:
    for mode in ('original', 'cudnn_deterministic', 'deterministic_algorithms'):
        row = dict(mode=mode, results=[])
        try:
            torch.use_deterministic_algorithms(mode == 'deterministic_algorithms')
            with torch.backends.cudnn.flags(enabled=original['enabled'], benchmark=original['benchmark'],
                                            deterministic=mode != 'original', allow_tf32=original['allow_tf32']):
                reference = None
                for repeat in range(3):
                    conv.zero_grad(set_to_none=True)
                    x.grad = None
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = conv(x)
                    output.backward(upstream)
                    current = conv.weight.grad.detach().cpu().clone()
                    if reference is None:
                        reference = current
                    delta = current.double() - reference.double()
                    row['results'].append(dict(repeat=repeat, max_abs=float(delta.abs().max()),
                                               different=int(torch.count_nonzero(delta)), total=delta.numel()))
                    del output, current, delta
        except Exception as exc:
            row['error'] = repr(exc)
        receipt['modes'].append(row)
finally:
    torch.use_deterministic_algorithms(original['deterministic_algorithms'])
receipt['finished'] = time.time()
receipt['status'] = 'completed_kernel_repeatability_only'
receipt['acceptance_limit'] = 'Synthetic native operator reproduction; does not establish PPO output tolerance, kernel binding or production speed.'
AUDIT.mkdir(parents=True, exist_ok=True)
(AUDIT / 'native-conv-repeatability.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
