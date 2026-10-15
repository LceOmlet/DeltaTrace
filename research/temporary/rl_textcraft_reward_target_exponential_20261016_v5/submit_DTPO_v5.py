"""Reuse the protected original formal launcher after explicit v4 withdrawal.

The previous DT is already stopped; no stop/restart/kill operation is issued.
"""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
original=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/submit_DTPO_variance_v3.py'
code=original.read_text()
replacements={
    "OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'":"OUT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
    "PREPARED=OUT/'DTPO-variance-v3-prepared'":"PREPARED=OUT/'v5-prepared'",
    'DTPO-variance-v3-job.json':'DTPO-v5-job.json',
    "old=json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())":
        "old=next(j for j in json.loads((ROOT/'formal-training.json').read_bytes())['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='dt')\nold=dict(old,run=str(Path(old['output']).parent),birth=old['observed_process_created_unix'])\nassert old['version']=='textcraft-dt-weighted-reward-targets-20261015-v4'",
    "ppo=json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())":
        "ppo=json.loads((ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json').read_bytes())",
    "protected_identities=[(3174448,1791793629.73),(ppo['pid'],ppo['birth'])]":
        "protected_identities=[(ppo['pid'],ppo['birth'])]\nfor pid,birth in ((3953367,1792041179.18),(3955296,1792041198.10),(2617231,1791949058.07),(2617232,1791949058.07)):\n    protected_identities.append((pid,birth))",
    'before-v3-launch-mx-smi.txt':'before-v5-launch-mx-smi.txt',
    'submitted_v3_initialization_pending':'submitted_v5_initialization_pending',
    'before-v3.json':'before-v5.json',
    "hourly_automation='PAUSED'":"hourly_automation='ACTIVE'",
}
for old,new in replacements.items():
    assert old in code,old;code=code.replace(old,new)
start=code.index("stop=PREPARED/'previous-DTPO-stop.json'")
end=code.index('for identity in protected:assert psutil.Process',start)
code=code[:start]+'''withdrawal=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/DTPO-v4-method-withdrawn.json'
withdrawn=json.loads(withdrawal.read_bytes())
assert withdrawn['status']=='stopped_method_equivalence_not_established' and not withdrawn['survivors']
assert (old['pid'],old['birth'])==(withdrawn['job']['pid'],withdrawn['job']['birth'])
assert not psutil.pid_exists(old['pid']) or psutil.Process(old['pid']).create_time()!=old['birth']
'''+code[end:]
assert 'os.killpg(' not in code and '.kill()' not in code
exec(compile(code,str(original),'exec'),globals())
