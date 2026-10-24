"""Bind the existing collector and plotting owners to independent v11."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
owner = HERE.parent/'rl_textcraft_v5_ssm_fp32_formal_20261022_v1/plot_algorithms.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '1b21277b2c50c41407bb9bdf3a31bbda9d13633b0c5b46fec678c6f79ab719fe'
source = owner.read_text(encoding='utf-8')
for before, after in [
    ('textcraft-v5-native-SSM-fp32-20261022-v1', 'textcraft-v13-empty-target-grpo-whiten-20261024-v1'),
    ('textcraft-dtpo-v5-ssm-fp32-8gpu-formal-20261022-v1', 'textcraft-dtpo-v13-empty-target-grpo-whiten-ssm-fp32-8gpu-formal-20261024-v1'),
    ('textcraft-v5-ssm-fp32-eight-gpu-vs-historical-PPO-GRPO-20261022-v1', 'textcraft-v13-empty-target-grpo-whiten-eight-gpu-vs-historical-PPO-GRPO-20261024-v1'),
    ('v5（本次八卡，SSM FP32）', 'v13（八卡，无target回退＋gu归一化）'),
    ("('Current eight-GPU v5', 'Current eight-GPU v5')", "('Current eight-GPU v5', 'Current eight-GPU v13')"),
    ('本次v5', '本次v13'),
    ('Current v5 and independent', 'Current v13 and independent'),
    ('all_completed_iterations_of_current_v5', 'all_completed_iterations_of_current_v13'),
    ('algorithm-comparison-v5-complete', 'algorithm-comparison-v13-complete'),
    ('v5 / PPO / GRPO', 'v13 / PPO / GRPO'),
]:
    assert before in source, before
    source = source.replace(before, after)
exec(compile(source, str(owner), 'exec'), globals())
