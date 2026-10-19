"""Keep measured-global DTPO and explicitly estimated archived control norms."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1/plot_historical_controls.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='3422f7cf7e4cef2b047a72f42e6e92273fd458b5c571260aa7f9ed92495d9c29'
code=owner.read_text(encoding='utf-8').replace('textcraft-eight-gpu-v9-clip-u-20261019-v2',
    'textcraft-eight-gpu-v9-negative-clip2-20261019-v3')
code=code.replace('textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2',
    'textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v3')
code=code.replace('textcraft-v9-clip-u-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v2',
    'textcraft-v9-negative-clip2-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v3')
code=code.replace('DTPO v9 clip(u)（8卡）','DTPO v9 u∈[-2,1]（8卡）')
exec(compile(code,str(owner),'exec'),globals())
