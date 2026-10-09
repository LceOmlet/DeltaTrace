"""Inspect existing AppWorld memory-owner and accepted numerical imports on CPU."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch


modules = {}
for name in ('qwen35_dense_finite_runner', 'qwen35_gdn_finite',
             'qwen35_decoder_finite', 'qwen35_answer_finite',
             'vendor_fa_finite_bf16_d256', 'deltatrace_rollout', 'verl.workers.fsdp_workers'):
    module = importlib.import_module(name)
    path = Path(module.__file__)
    modules[name] = dict(path=str(path), resolved=str(path.resolve()),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())

assert modules['qwen35_dense_finite_runner']['sha256'] == '7d6f57f61ecde7ce3506b8ef58d61268859fc04357c35de99a1892928c4ba7a8'
assert modules['qwen35_gdn_finite']['sha256'] == '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1'
assert modules['verl.workers.fsdp_workers']['sha256'] == 'e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39'
config = Path(os.environ['DT_ENVIRONMENT_JSON'])
assert json.loads(config.read_bytes())['qwen35']['dt_offload_replay_mixer'] is True
assert not torch.cuda.is_initialized()
print(json.dumps(dict(modules=modules,
    environment_config=dict(path=str(config), sha256=hashlib.sha256(config.read_bytes()).hexdigest()),
    dt_offload_replay_mixer=True, cuda_initialized=False, model_DT_optimizer_calls=0,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Fresh CPU actual imports/config only. Composed capacity, QVA or training is not asserted.')))
