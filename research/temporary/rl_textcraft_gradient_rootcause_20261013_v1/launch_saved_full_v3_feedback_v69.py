"""Use the existing free-device-checked diagnostic launcher for full v3."""
from pathlib import Path

source = Path(__file__).with_name('launch_common_update_feedback_v47.py')
text = source.read_text()
changes = {
    'native-common-update-feedback-v47': 'native-full-current-v3-feedback-v69',
    'common-update-feedback-source-v47.json': 'full-current-v3-feedback-source-v69.json',
    'common-update-feedback-v47': 'full-current-v3-feedback-v69',
    'observe_common_update_feedback_v47.py': 'observe_saved_full_v3_feedback_v69.py',
    '[(1419849,1791977138.31),(3174448,1791793629.73)]':
        '[(3140624,1791993570.92),(3174448,1791793629.73)]',
    "launch_path=Path(spec['original_launch'])": "launch_path=Path(spec['current_v3_launch'])",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
