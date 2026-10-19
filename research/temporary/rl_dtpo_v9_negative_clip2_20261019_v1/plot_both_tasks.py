"""Adapt only identity and label; preserve the existing collector and renderer."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1/plot_both_tasks.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='d97bde2b4d0251d6f058060d1b9f9ac8d55ca9009b6cb481e07b615c762860fc'
code=owner.read_text(encoding='utf-8').replace('-eight-gpu-v9-clip-u-20261019-v2',
    '-eight-gpu-v9-negative-clip2-20261019-v3')
code=code.replace('DTPO v9 clip(u)·std(g·u)（8卡）','DTPO v9 u∈[-2,1]·std(g·u)（8卡）')
exec(compile(code,str(owner),'exec'),globals())
