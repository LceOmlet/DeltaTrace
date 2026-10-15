"""Prepare the existing native PPO pause only at its next complete checkpoint.

This preserves the progress made since the last restore; it is a scheduling
boundary, not an added training-quality or mathematical acceptance condition.
"""
from pathlib import Path

source = Path(__file__).with_name('prepare_PPO_pause_plan_v70.py')
code = source.read_text()
changes = {
    'assert step >= 20,': 'assert step >= 30,',
    "torch.load(files['optim'], map_location='cpu', weights_only=False)":
        "torch.load(files['optim'], map_location='cpu', weights_only=False, mmap=True)",
    'PPO-native-pause-for-v69-plan-v70.json': 'PPO-native-pause-for-same-session-v113-plan-v114.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
