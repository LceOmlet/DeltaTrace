"""Compare dtype/normalization through the pinned test's own options.

The original test, random inputs, FP32 reference and assertions are unchanged.
The recorded dtype and use_qk_l2norm_in_kernel fields identify each case.
Production is unchanged; no operator, reference, or tolerance is reimplemented.
"""
import json
import argparse
import os
from pathlib import Path
import time
import torch
import fla.utils
from verify_official_kernel_tolerances import load

parser=argparse.ArgumentParser()
parser.add_argument('--dtype',choices=('bfloat16','float16'),default='bfloat16')
parser.add_argument('--fused-norm',action='store_true')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
r = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts'
owner = load('pinned_fla_test', r/'training-setup/official-kernel-tests/test_gated_delta_v041.py')
out = args.output or r/'rollout-major-cost/native-fla-normalization-isolation.json'
assert not fla.utils.FLA_CI_ENV
torch.set_num_threads(8)
result = dict(scope=__doc__, cases=[])
for length in (128, 447):
    started = time.perf_counter()
    case = dict(B=1, T=length, H=32, D=128, scale=128**-.5,
                gate_logit_normalizer=1., mask_p=0., use_qk_l2norm_in_kernel=args.fused_norm)
    try:
        owner.test_chunk(**case, dtype=getattr(torch,args.dtype))
        case['status'] = 'passed'
    except AssertionError as exc:
        case.update(status='failed', error=str(exc))
    case['seconds'] = time.perf_counter()-started
    case['dtype'] = args.dtype
    result['cases'].append(case)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(case),flush=True)
