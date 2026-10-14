"""Reuse original full-restore verification at the actual planned checkpoint."""
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
planned = json.loads((OUT / 'PPO-native-pause-for-v69-plan-v70.json').read_bytes())
restore_step = planned['completed_iteration']
source = Path(__file__).with_name('record_native_PPO_restore_v59.py')
text = source.read_text()
changes = {
    'native-PPO-resume-after-v47-v55': 'native-PPO-resume-after-v69-v70',
    'native-PPO-bindings-v56.json': 'native-PPO-bindings-v70.json',
    "==15 and role['optimizer_state_steps']==[60.]": "==restore_step and role['optimizer_state_steps']==[float(4*restore_step)]",
    'scheduler=15,optimizer_step=60': 'scheduler=restore_step,optimizer_step=4*restore_step',
    "status='native_PPO_full_checkpoint15_restore_verified'": "status=f'native_PPO_full_checkpoint{restore_step}_restore_verified'",
    'resume_native_PPO_after_diagnostic_v55.py': 'resume_PPO_after_saved_v3_v70.py',
    'native-PPO-restore-verification-v59.json': 'native-PPO-restore-verification-v70.json',
    'resume-v55-job.json': 'resume-v70-job.json',
    'before-verified-restore-v59.json': 'before-verified-restore-v70.json',
    'checkpoint15.': 'restored checkpoint.',
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
