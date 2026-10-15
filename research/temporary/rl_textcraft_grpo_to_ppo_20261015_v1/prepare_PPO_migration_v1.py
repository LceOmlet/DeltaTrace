"""Prepare the existing native full PPO checkpoint for user-requested GPU migration."""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/prepare_PPO_pause_plan_v70.py'
code = source.read_text()
changes = {
    "assert step >= 20": "assert step >= 30",
    "torch.load(files['optim'], map_location='cpu', weights_only=False)":
        "torch.load(files['optim'], map_location='cpu', weights_only=False, mmap=True)",
    "for role, identity in (('DTPO_v3', dt_job), ('GRPO', grpo)):":
        "for role, identity in (('DTPO_v3', dt_job),):",
    'PPO-native-pause-for-v69-plan-v70.json': 'PPO-native-pause-for-GPU45-migration-v117.json',
    'Reuse the user-authorized temporary PPO pause for a finite gradient diagnostic, then restore the same original run through native full-checkpoint loading.':
        'User requested stopping GRPO and continuing the existing PPO on released GPU4/5 through the same native full checkpoint; no diagnostic or algorithm change.',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
