"""Reuse the exact existing process-group stop after its full-restore plan exists."""
import json
from pathlib import Path
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
planned = json.loads((OUT / 'PPO-native-pause-for-v69-plan-v70.json').read_bytes())
prepared = json.loads((OUT / 'native-PPO-resume-after-v69-v70-prepared.json').read_bytes())
assert planned['status'] == 'planned_only_PPO_still_running'
assert prepared['checkpoint'] == planned['checkpoint']
assert set(prepared['configuration_changes']) == {'trainer.resume_mode', 'trainer.resume_from_path'}
current = json.loads((OUT / 'native-PPO-control-v9-job.json').read_bytes())
assert (current['pid'], current['birth']) == (planned['job']['pid'], planned['job']['birth'])
assert Path(planned['marker']).read_text().strip() == str(planned['completed_iteration'])
live_protected = []
for identity in planned['protected_jobs']:
    try:
        p = psutil.Process(identity['pid'])
        alive = p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        alive = False
    if identity['required_alive']:
        assert alive
    if alive:
        live_protected.append(identity)
source = Path(__file__).with_name('stop_PPO_for_authorized_diagnostic_v54.py')
text = source.read_text()
changes = {
    'PPO-temporary-stop-for-v47-v54.json': 'PPO-temporary-stop-for-v69-v70.json',
    '[(1419849,1791977138.31),(3174448,1791793629.73),(2614045,1791949034.31)]':
        "[(x['pid'],x['birth']) for x in live_protected]",
    'User explicitly chose temporary PPO stop and full native restore after the bounded diagnostic.':
        'User previously authorized temporary PPO pause for finite gradient diagnostics and full native restore; this is the same workflow at the latest verified native checkpoint.',
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
