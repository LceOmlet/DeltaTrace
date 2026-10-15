"""Record actual native actor/critic restore and execution provenance after migration."""
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
planned = json.loads((OUT/'PPO-native-pause-for-GPU45-migration-v117.json').read_bytes())
restore_step = planned['completed_iteration']
source = OUT/'record_native_PPO_restore_v59.py'
code = source.read_text()
changes = {
    'native-PPO-resume-after-v47-v55': 'native-PPO-resume-GPU45-v117',
    'native-PPO-bindings-v56.json': 'native-PPO-bindings-v117.json',
    "==15 and role['optimizer_state_steps']==[60.]":
        "==restore_step and role['optimizer_state_steps']==[float(4*restore_step)]",
    'scheduler=15,optimizer_step=60': 'scheduler=restore_step,optimizer_step=4*restore_step',
    "status='native_PPO_full_checkpoint15_restore_verified'":
        "status=f'native_PPO_full_checkpoint{restore_step}_restore_verified'",
    'resume_native_PPO_after_diagnostic_v55.py': 'resume_PPO_on_GPU45_v1.py',
    'native-PPO-restore-verification-v59.json': 'native-PPO-restore-verification-v117.json',
    'resume-v55-job.json': 'resume-GPU45-v117-job.json',
    'before-verified-restore-v59.json': 'before-verified-restore-v117.json',
    'checkpoint15.': 'restored checkpoint.',
    "rootcause_goal_status='active'": "rootcause_goal_status='paused'",
    "hourly_automation='PAUSED'": "hourly_automation='ACTIVE_PLOTTING_ONLY'",
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
