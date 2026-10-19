"""Compose the original real DT/B4/32768 checks for the negative2 coefficient."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/run_native_behavior_v9.py'
code=owner.read_text().replace('dtpo-v9-clip-u-postgroup-variance-20261019-v1',
    'dtpo-v9-negative-clip2-postgroup-variance-20261019-v3')
code=code.replace('.clamp(-1,1)','.clamp(-2,1)')
anchor="assert kwargs.get('clip_exponential_credit') is True"
assert code.count(anchor)==1
code=code.replace(anchor,anchor+";assert kwargs.get('exponential_credit_negative_clip')==2.0")
anchor='exec(compile(code,str(source),\'exec\'),globals())'
assert code.count(anchor)==1
code=code.replace(anchor,"code=code.replace(\"'PYTORCH_CUDA_ALLOC_CONF')\",\"'PYTORCH_CUDA_ALLOC_CONF','DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP')\")\n"+anchor)
exec(compile(code,str(owner),'exec'),globals())
