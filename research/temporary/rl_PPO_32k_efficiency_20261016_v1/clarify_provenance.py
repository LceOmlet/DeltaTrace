"""Clarify byte SHA versus normalized source-text SHA in the preparation receipt."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
BASE=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
prepared=json.loads((AUDIT/'prepared-candidate.json').read_bytes())
entries=[]
for item in prepared['changes']:
    relative=Path(item['path']).relative_to(ROOT/'candidates/PPO-32k-native-efficiency-20261016-v1/verl-owner')
    original=BASE/relative
    entries.append(dict(path=str(original),actual_bytes_SHA256=hashlib.sha256(original.read_bytes()).hexdigest(),
        normalized_text_SHA256=hashlib.sha256(original.read_text().encode()).hexdigest(),
        prepared_receipt_before_field=item['before']))
record=dict(scope=__doc__,sources=entries,
    clarification='Initial before field hashed Python read_text normalized line endings. Actual imported file identity is the byte SHA. Existing receipt retained; preparation script now records bytes.',
    CPU_preflight_failure=dict(type='ImportError',detail='AdvantageEstimator belongs to original ray_trainer, not core_algos. Import corrected before any Ray/model/optimizer call.'),
    candidate_status='prepared and native tiny-handler passed; full critic verification pending',formal_mutations=0)
(AUDIT/'provenance-clarification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
