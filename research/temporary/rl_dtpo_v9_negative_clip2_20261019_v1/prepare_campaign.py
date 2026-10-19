"""Compose the original fresh six-run configuration for negative2 identities."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/prepare_campaign.py'
code=owner.read_text().replace('dtpo-v9-clip-u-postgroup-variance-20261019-v1',
    'dtpo-v9-negative-clip2-postgroup-variance-20261019-v3')
code=code.replace('dtpo-v9-clip-u','dtpo-v9-negative-clip2')
code=code.replace('eight-gpu-v9-clip-u','eight-gpu-v9-negative-clip2')
code=code.replace('20261019-v1','20261019-v3')
code=code.replace('clipped-minus1-plus1-v9','clipped-minus2-plus1-v9')
anchor="                row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = candidate['credit_mode']"
assert code.count(anchor)==1
code=code.replace(anchor,anchor+"\n                for key in ('PYTORCH_CUDA_ALLOC_CONF','DT_NATIVE_HOST_CACHE_REPLAY_LAYER','DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP'):\n                    row['options']['+ray_init.runtime_env.env_vars.'+key] = candidate['effective_environment'][key]")
exec(compile(code,str(owner),'exec'),globals())
