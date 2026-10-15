"""Reuse the existing protected native launcher and unchanged v99 observer."""
from pathlib import Path

source = Path(__file__).with_name('launch_saved_update_natural_PG_v99.py')
code = source.read_text()
old = 'native-saved-update-natural-PG-v99'
new = 'native-same-session-saved-update-feedback-v113'
assert old in code
code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
