"""Reuse the original one-shot observer on the actual v3 workers."""
from pathlib import Path

source = Path(__file__).with_name("live_update_snapshot_v61.py")
text = source.read_text()
changes = {
    "textcraft-dt-grpo-multiplier-no-whiten-20261014-v1": "textcraft-dt-grpo-multiplier-variance-only-20261014-v3",
    "live-native-update-input-v61": "live-native-update-input-variance-v3",
    "DT-group-multiplier-v40-job.json": "DTPO-variance-v3-job.json",
    "WORKERS = {1426979: 1791977192.37, 1429032: 1791977210.63}":
        "WORKERS = {3151306: 1791993626.33, 3153050: 1791993643.98}",
    "approved zero-group route and no final whitening": "approved zero-group route and variance-only scaling without mean subtraction",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), "exec"), globals())
