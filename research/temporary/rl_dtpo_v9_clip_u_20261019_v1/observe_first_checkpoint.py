"""Reuse the original passive native-checkpoint observer for TextCraft v9/8 ranks."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/source/original_checkpoint_observer.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '497525da9922785991bee131aea5ca0f9f840b8aa4c50ab7163b9a7cb4b3c3bc'
code = owner.read_text()
replacements = [
    ('dtpo-exponential-credit-variance-20261017-v6','dtpo-v9-clip-u-postgroup-variance-20261019-v2'),
    ("CAMPAIGN = ROOT / 'receipts/appworld-formal-training-20261017-v1'",
     "CAMPAIGN = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'"),
    ("'source/verify_completed_checkpoint.py'","'source/verify_eight_rank_checkpoint.py'"),
    ("d5a6f3ee52e822356e5cb4d3df73215f3818ede9e033eda9e4b53719636ae865",
     "f55f07b5d1f7c962492e5118f99e5fcd1a56dadf508df2f5a8a987af304ddbe3"),
    ('native-AppWorld-PPO-checkpoint2','native-TextCraft-DT-checkpoint5'),
    ("j.get('task') == 'AppWorld' and j.get('method') == 'ppo'",
     "j.get('task') == 'TextCraft' and j.get('method') == 'dt'"),
    ("str(VERIFIER), 'ppo'", "str(VERIFIER), 'TextCraft', 'dt'"),
    ('first_checkpoint_step_matches_save2=(step == 2)', 'first_checkpoint_step_matches_save5=(step == 5)'),
    ('live native PPO marker','live native TextCraft DT marker'),
    ('The original\ntrainer writes loop_sampler.pt before its completed-checkpoint marker.',
     'The original\ntrainer completes role/data artifacts before its completed-checkpoint marker.'),
]
for old,new in replacements:
    assert old in code, old
    code=code.replace(old,new)
anchor = "        source = json.loads(Path(job['source_receipt']).read_bytes())\n"
assert code.count(anchor) == 1
code = code.replace(anchor, "        assert job['run']==str(ROOT/'runs/textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2')\n" + anchor)
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code,str(owner),'exec'),globals())
