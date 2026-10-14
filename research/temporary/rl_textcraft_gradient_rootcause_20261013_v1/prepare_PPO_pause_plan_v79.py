"""Use the existing full native checkpoint workflow at completed PPO25 or later."""
from pathlib import Path

source = Path(__file__).with_name('prepare_PPO_pause_plan_v70.py')
text = source.read_text()
changes = {
    "assert step >= 20,": "assert step >= 25,",
    "torch.load(files['optim'], map_location='cpu', weights_only=False)":
        "torch.load(files['optim'], map_location='cpu', weights_only=False, mmap=True)",
    'PPO-native-pause-for-v69-plan-v70.json': 'PPO-native-pause-for-control-v78-plan-v79.json',
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
