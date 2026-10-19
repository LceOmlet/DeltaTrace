"""Retain the requested measured-global v9 versus archived PPO/GRPO plot."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_textcraft_v5_eight_gpu_20261019_v1/plot_hourly.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='0b6da9c6e4e59ea8b05d7fce7eecc8d92a75e3d478ce2dff87bddef25aa730fb'
code=owner.read_text(encoding='utf-8')
code=code.replace('textcraft-v5-eight-gpu-20261019-v2','textcraft-eight-gpu-v9-clip-u-20261019-v2')
code=code.replace('textcraft-dtpo-v5-exponential-8gpu-200-20261019-v2','textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2')
code=code.replace('textcraft-v5-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v2',
    'textcraft-v9-clip-u-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v2')
code=code.replace('DTPO v5（新8卡）','DTPO v9 clip(u)（8卡）').replace('Current eight-GPU v5','Current eight-GPU v9')
code=code.replace('新八卡v5为全局实测','八卡v9为全局实测')
exec(compile(code,str(owner),'exec'),globals())
