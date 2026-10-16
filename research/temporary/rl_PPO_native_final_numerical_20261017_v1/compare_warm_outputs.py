"""Apply the pinned owner output assertion to the already completed warm cases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/PPO-native-final-numerical-20261017-v1'
if '--native' not in sys.argv:
    prepared = json.loads((ROOT / 'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json').read_bytes())
    env = dict(os.environ)
    env.update(prepared['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--native'], env=env))

import torch

status = json.loads((AUDIT / 'status.json').read_bytes())
assert status['status'] == 'completed_native_unobserved_update_output_replay'
assert all(case['full_preupdate_state_restore_exact'] and case['input_fields_exact'] for case in status['completed'])
native = torch.load(AUDIT / 'native_repeat-post-values.pt', weights_only=False, map_location='cpu')
candidate = torch.load(AUDIT / 'native_async-post-values.pt', weights_only=False, map_location='cpu')
assert torch.isfinite(native).all() and torch.isfinite(candidate).all()
result = dict(collected=time.time(), source_commit=status['source_commit'],
              reference='native_repeat, after two native optimizer steps in this same process',
              candidate='native_async, same full state and input, already initialized runtime',
              owner_test='Pinned VERL activation-offload test initializes with an update before reference/reload comparison',
              shape=list(native.shape), dtype=str(native.dtype),
              max_abs=float((native-candidate).abs().max()), different=int(torch.count_nonzero(native-candidate)),
              total=native.numel(), criterion=dict(atol=0,rtol=0,quantity='Native critic post-update values'),
              first_call_comparisons_preserved=status['comparisons'],
              during_update_gradient_or_optimizer_hooks=False,
              model_calls=0, optimizer_updates=0, checkpoint_create_calls=0,
              script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    torch.testing.assert_close(native, candidate, atol=0, rtol=0)
    result['status'] = 'passed_warm_native_owner_output_criterion'
except AssertionError as exc:
    result['status'] = 'failed_warm_native_owner_output_criterion'
    result['error'] = str(exc)
(AUDIT / 'warm-native-output-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
