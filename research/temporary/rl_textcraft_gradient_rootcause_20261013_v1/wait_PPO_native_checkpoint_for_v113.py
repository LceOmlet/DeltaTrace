"""Reuse the read-only native checkpoint waiter for the current PPO identity."""
from pathlib import Path

source = Path(__file__).with_name('wait_PPO_native_checkpoint_v82.py')
code = source.read_text()
changes = {
    '(1308636, 1792016095.73)': '(2666718, 1792028884.24)',
    'verified-PPO-native-checkpoint-wait-v82.json': 'verified-PPO-native-checkpoint-wait-v113-v114.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
