"""Reuse the accepted v5 collector and historical-control renderer."""
from pathlib import Path

owner = Path(__file__).resolve().parents[1]/'rl_textcraft_v5_eight_gpu_20261019_v1/plot_hourly.py'
code = owner.read_text(encoding='utf-8')
code = code.replace('textcraft-v5-eight-gpu-20261019-v1', 'textcraft-v5-credit-integration-20261020-v1')
code = code.replace('textcraft-dtpo-v5-exponential-8gpu-200-20261019-v1',
    'textcraft-dtpo-v5-exponential-verified-8gpu-200-20261020-v1')
code = code.replace('textcraft-v5-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v1',
    'textcraft-v5-verified-eight-gpu-historical-PPO-GRPO-global-norm-20261020-v1')
code = code.replace('DTPO v5（新8卡）', 'DTPO v5（8卡，20261020）')
exec(compile(code, str(owner), 'exec'), globals())
