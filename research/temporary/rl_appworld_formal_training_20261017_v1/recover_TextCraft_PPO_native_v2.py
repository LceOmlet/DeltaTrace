"""Reuse the native full-checkpoint recovery for the second TextCraft PPO OOM."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-formal-training-20261017-v1/source/recover_TextCraft_PPO_native.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8318007336e45a6ab102b500d8b1cdbc3a151c9b682d3ad877cf84279fa6fc2e'
code=source.read_text()
replacements={
    'textcraft-PPO-global-OOM-recovery-20261017-v1':'textcraft-PPO-global-OOM-recovery-20261017-v2',
    'TextCraft-PPO-native-OOM-recovery-input.json':'TextCraft-PPO-native-OOM-recovery-input-v2.json',
    'TextCraft-PPO-native-global-OOM-recovery-v1':'TextCraft-PPO-native-global-OOM-recovery-v2',
    'TextCraft-PPO-global-OOM-exit.json':'TextCraft-PPO-global-OOM-third-exit.json',
    '-global-OOM-v1.':'-global-OOM-v2.',
    'before-global-OOM-recovery-v1.json':'before-global-OOM-recovery-v2.json',
    'resume-global-OOM-recovery-v1-job.json':'resume-global-OOM-recovery-v2-job.json',
    '[(878971,1792091830.03),(1395665,1792177213.42)]':
        '[(878971,1792091830.03),(2086772,1792183876.72),(2477154,1792187585.22)]',
    'TextCraft_PPO_global_OOM_recovery_20261017_v1':'TextCraft_PPO_global_OOM_recovery_20261017_v2',
}
for old,new in replacements.items():
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'),globals())
