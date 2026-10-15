"""Reuse the protected, checkpoint-preserving formal DTPO replacement owner."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
original=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/submit_DTPO_variance_v3.py'
code=original.read_text()
replacements={
    "OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'":
        "OUT=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4'",
    "PREPARED=OUT/'DTPO-variance-v3-prepared'":"PREPARED=OUT/'v4-prepared'",
    "DTPO-variance-v3-job.json":"DTPO-v4-job.json",
    "old=json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())":
        "old=next(j for j in json.loads((ROOT/'formal-training.json').read_bytes())['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='dt')\nold=dict(old,run=str(Path(old['output']).parent),birth=old['observed_process_created_unix'])\nassert (old['pid'],old['birth'])==(3140624,1791993570.92)",
    "ppo=json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())":
        "ppo=json.loads((ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json').read_bytes())",
    "protected_identities=[(3174448,1791793629.73),(ppo['pid'],ppo['birth'])]":
        "protected_identities=[(ppo['pid'],ppo['birth'])]\nfor pid in (3953367,3955296,2617231,2617232):\n    q=psutil.Process(pid);assert q.status()!=psutil.STATUS_ZOMBIE\n    protected_identities.append((pid,q.create_time()))",
    'previous-DTPO-stop.json':'previous-DTPO-v3-stop.json',
    'User requests fresh DTPO v3 restart with variance-only normalization.':
        'User requests verified fresh DTPO v4 with reward-settlement targets and weighted linear finite credit.',
    'stopped_for_user_requested_fresh_v3':'stopped_for_user_requested_fresh_v4',
    'before-v3-launch-mx-smi.txt':'before-v4-launch-mx-smi.txt',
    'submitted_v3_initialization_pending':'submitted_v4_initialization_pending',
    'before-v3.json':'before-v4.json',
    "hourly_automation='PAUSED'":"hourly_automation='ACTIVE'",
}
for old,new in replacements.items():
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(original),'exec'),globals())
